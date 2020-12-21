__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["PairwiseSampler"]

from reckit import DataIterator
from reckit import randint_choice
from reckit import typeassert
from reckit import pad_sequences
from collections import Iterable
from collections import OrderedDict
import numpy as np
from scipy.sparse import csr_matrix
from utils.tools import csr_to_user_dict


class Sampler(object):
    """Base class for all sampler to sample negative items.
    """

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


@typeassert(user_pos_dict=dict)
def _generate_positive_items(user_pos_dict):
    if not user_pos_dict:
        raise ValueError("'user_pos_dict' cannot be empty.")

    users_list, items_list = [], []
    user_n_pos = OrderedDict()

    for user, items in user_pos_dict.items():
        items_list.append(items)
        users_list.append(np.full_like(items, user))
        user_n_pos[user] = len(items)
    users_arr = np.concatenate(users_list)
    items_arr = np.concatenate(items_list)
    return user_n_pos, users_arr, items_arr


@typeassert(user_pos_dict=dict, len_seqs=int, len_next=int, pad=(int, None))
def _generative_time_order_positive_items(user_pos_dict, len_seqs=1, len_next=1, pad=None):
    if not user_pos_dict:
        raise ValueError("'user_pos_dict' cannot be empty.")

    users_list, item_seqs_list, next_items_list = [], [], []
    user_n_pos = OrderedDict()

    tot_len = len_seqs + len_next
    for user, seq_items in user_pos_dict.items():
        if isinstance(seq_items, np.ndarray):
            seq_items = np.array(seq_items, dtype=np.int32)
        if len(seq_items) >= tot_len:
            user_n_pos[user] = 0
            for idx in range(len(seq_items)-tot_len+1):
                tmp_seqs = seq_items[idx:idx+tot_len]
                item_seqs_list.append(tmp_seqs[:len_seqs].reshape([1, len_seqs]))
                next_items_list.append(tmp_seqs[len_seqs:].reshape([1, len_next]))
                users_list.append(user)
                user_n_pos[user] += 1
        elif len(seq_items) > len_next and pad is not None:  # padding

            next_items_list.append(seq_items[-len_next:].reshape([1, len_next]))
            tmp_seqs = pad_sequences([seq_items[:-len_next]], value=pad, max_len=len_seqs,
                                     padding='pre', truncating='pre', dtype=np.int32)
            item_seqs_list.append(tmp_seqs.squeeze().reshape([1, len_seqs]))
            users_list.append(user)
            user_n_pos[user] = 1
        else:  # discard
            continue
    users_arr = np.int32(users_list)
    item_seqs_arr = np.concatenate(item_seqs_list).squeeze()
    next_items_arr = np.concatenate(next_items_list).squeeze()
    return user_n_pos, users_arr, item_seqs_arr, next_items_arr


@typeassert(user_n_pos=OrderedDict, num_neg=int, num_items=int, user_pos_dict=dict)
def _sampling_negative_items(user_n_pos, num_neg, num_items, user_pos_dict):
    if num_neg <= 0:
        raise ValueError("'neg_num' must be a positive integer.")

    neg_items_list = []
    for user, n_pos in user_n_pos.items():
        neg_items = randint_choice(num_items, size=n_pos*num_neg, exclusion=user_pos_dict[user])
        if num_neg == 1:
            neg_items = neg_items if isinstance(neg_items, Iterable) else [neg_items]
            neg_items_list.append(neg_items)
        else:
            neg_items = np.reshape(neg_items, newshape=[n_pos, num_neg])
            neg_items_list.append(neg_items)

    return np.concatenate(neg_items_list)


class PairwiseSampler(Sampler):
    """Sampling negative items and construct pairwise training instances.

    The training instances consist of `batch_user`, `batch_pos_item` and
    `batch_neg_items`, where `batch_user` and `batch_pos_item` are lists
    of users and positive items with length `batch_size`, and `neg_items`
    does not interact with `user`.

    If `neg_num == 1`, `batch_neg_items` is also a list of negative items
    with length `batch_size`;  If `neg_num > 1`, `batch_neg_items` is an
    array like list with shape `(batch_size, neg_num)`.
    """

    @typeassert(dataset=csr_matrix, num_neg=int, batch_size=int, shuffle=bool, drop_last=bool)
    def __init__(self, dataset, num_neg=1, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `PairwiseSampler` instance.

        Args:
            dataset (csr_matrix): An instance of `csr_matrix`.
            num_neg (int): How many negative items for each positive item.
                Defaults to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(PairwiseSampler, self).__init__()
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_neg = num_neg
        self.num_items = dataset.shape[1]

        self.user_pos_dict = csr_to_user_dict(dataset)

        self.user_n_pos, self.all_users, self.pos_items = \
            _generate_positive_items(self.user_pos_dict)

    def __iter__(self):
        neg_items = _sampling_negative_items(self.user_n_pos, self.num_neg,
                                             self.num_items, self.user_pos_dict)

        data_iter = DataIterator(self.all_users, self.pos_items, neg_items,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle, drop_last=self.drop_last)
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_pos_items), np.asarray(bat_neg_items)

    def __len__(self):
        n_sample = len(self.all_users)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size
