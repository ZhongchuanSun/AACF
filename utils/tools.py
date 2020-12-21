import numpy as np
from scipy.sparse import csr_matrix
from reckit import typeassert
from reckit import randint_choice


@typeassert(matrix=csr_matrix)
def csr_to_user_dict(matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    idx_value_dict = {}
    for idx, value in enumerate(matrix):
        if len(value.indices) > 0:
            idx_value_dict[idx] = value.indices.copy()
    return idx_value_dict


@typeassert(matrix=csr_matrix, neg_num=int)
def csr_to_pairwise(matrix, neg_num=1):
    all_user_list, all_pos_list, all_neg_list = [], [], []
    user_num, item_num = matrix.shape

    for user in list(range(user_num)):
        pos_items = matrix[user].indices
        n_samples = len(pos_items)*neg_num
        if n_samples <= 0:
            continue
        neg_items = randint_choice(item_num, size=n_samples, replace=True, exclusion=pos_items)

        all_user_list.extend([user] * n_samples)
        pos_items = np.tile(pos_items, neg_num).reshape([-1])
        all_pos_list.extend(pos_items)
        all_neg_list.extend(neg_items)

    return all_user_list, all_pos_list, all_neg_list
