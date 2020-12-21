import numpy as np
from data.sampler import PairwiseSampler
from tqdm import tqdm


class DNSBPR(object):
    def __init__(self, num_users, num_items, num_dims):
        self.user_embeds = np.random.uniform(low=0, high=1.0, size=[num_users, num_dims]).astype(np.float32)
        self.item_embeds = np.random.uniform(low=0, high=1.0, size=[num_items, num_dims]).astype(np.float32)
        self.item_biases = np.random.uniform(low=0, high=1.0, size=num_items).astype(np.float32)

    def train_model(self, train_matrix, evaluator):
        try:
            from .cython.dnsbpr_func import dnsbpr_update
        except:
            raise ImportError("Please compile pretraining code with command: "
                              "'python setup.py build_ext --inplace'.")

        data_iter = PairwiseSampler(train_matrix, num_neg=15, batch_size=train_matrix.nnz, shuffle=False,
                                    drop_last=False)
        for _ in tqdm(range(300), desc="pretraining"):
        # for epoch in range(1000):
            user1d, pos_item1d, neg_items2d = list(data_iter)[0]
            dnsbpr_update(user1d, pos_item1d, neg_items2d, 0.01, 0.05,
                          self.user_embeds, self.item_embeds, self.item_biases)
            # print(f"{epoch}:\t{evaluator.evaluate(self)}")
            # if (epoch+1) % 100 == 0:
            #     np.save(f"./bprdns1_{epoch}.npy", self.parameters())

    def parameters(self):
        return self.user_embeds, self.item_embeds, self.item_biases

    def predict(self, users):
        return np.matmul(self.user_embeds[users], self.item_embeds.T) + self.item_biases
