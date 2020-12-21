from model.base import AbstractRecommender
import tensorflow as tf
from modules import log_loss
from reckit import DataIterator
from reckit import timer
import numpy as np
from utils.tools import csr_to_user_dict
import os
from modules import inner_product
from .DNSBPR import DNSBPR
from reckit import arg_top_k, batch_randint_choice


class Generator(object):
    def __init__(self, users_num, items_num, factors_num, params):
        self.user_embeddings = tf.Variable(params[0], name="user_embedding")
        self.item_embeddings = tf.Variable(params[1], name="item_embedding")
        self.item_biases = tf.Variable(params[2], name="item_bias")

    def parameters(self):
        return [self.user_embeddings, self.item_embeddings, self.item_biases]

    def forward(self, users, att_items):
        users_embedding = tf.nn.embedding_lookup(self.user_embeddings, users)  # (b, d)
        tf.add_to_collection("g_reg", tf.nn.l2_loss(users_embedding))
        users_embedding = tf.expand_dims(users_embedding, axis=1)  # (b, 1, d)

        att_embedding = tf.nn.embedding_lookup(self.item_embeddings, att_items)  # (b, n, d)
        tf.add_to_collection("g_reg", tf.nn.l2_loss(att_embedding))

        att_bias = tf.gather(self.item_biases, att_items)  # (b, n)
        tf.add_to_collection("g_reg", tf.nn.l2_loss(att_bias))

        dot_p = tf.matmul(users_embedding, att_embedding, transpose_b=True)  # (b, 1, n)
        att_logits = tf.squeeze(dot_p) + att_bias  # (b, n)
        att_prob = tf.nn.softmax(att_logits)

        return att_prob  # (b, n)


class Critic(object):
    def __init__(self, users_num, items_num, factors_num):
        self.user_embeddings = tf.Variable(tf.random_uniform([users_num, factors_num],
                                                             minval=-0.05, maxval=0.05), name="user_embedding")
        self.item_embeddings = tf.Variable(tf.random_uniform([items_num, factors_num],
                                                             minval=-0.05, maxval=0.05), name="item_embedding")
        self.item_biases = tf.Variable(tf.zeros([items_num]), name="item_bias")

    def parameters(self):
        return [self.user_embeddings, self.item_embeddings, self.item_biases]

    def forward(self, users, pos_items, att_items, att_weights):
        yi_hat = self._pos_predict(users, pos_items)
        yj_hat = self._att_predict(users, att_items, att_weights)
        loss = log_loss(yi_hat - yj_hat)
        return tf.reduce_mean(loss)

    def _pos_predict(self, users, items):
        user_embedding = tf.nn.embedding_lookup(self.user_embeddings, users)
        item_embedding = tf.nn.embedding_lookup(self.item_embeddings, items)
        item_bias = tf.gather(self.item_biases, items)
        logit = inner_product(user_embedding, item_embedding) + item_bias
        tf.add_to_collection("d_reg", tf.nn.l2_loss(user_embedding))
        tf.add_to_collection("d_reg", tf.nn.l2_loss(item_embedding))
        tf.add_to_collection("d_reg", tf.nn.l2_loss(item_bias))

        return logit

    def _att_predict(self, users, att_item, att_weight):
        u_embedding = tf.nn.embedding_lookup(self.user_embeddings, users)  # (b, d)
        item_embeddings = tf.nn.embedding_lookup(self.item_embeddings, att_item)  # (b, n, d)
        item_bias = tf.gather(self.item_biases, att_item)  # (b, n)

        # att_weight (b, n)
        i_bias = tf.reduce_sum(tf.multiply(att_weight, item_bias), axis=1)  # (b)

        # att_weight = tf.reshape(att_weight, [-1, 1, att_num])  # (b, 1, n)
        att_weight = tf.expand_dims(att_weight, axis=1)  # (b, 1, n)
        i_embedding = tf.squeeze(tf.matmul(att_weight, item_embeddings))  # (b, d)

        logit = inner_product(u_embedding, i_embedding) + i_bias
        tf.add_to_collection("d_reg", tf.nn.l2_loss(u_embedding))
        tf.add_to_collection("d_reg", tf.nn.l2_loss(i_embedding))
        tf.add_to_collection("d_reg", tf.nn.l2_loss(i_bias))

        return logit


def softmax(arr, tau=1.0):
    arr /= tau
    exp_arr = np.exp(arr)
    sum_exp = np.sum(exp_arr, axis=1, keepdims=True)
    return exp_arr/sum_exp


class AACF(AbstractRecommender):
    def __init__(self, config):
        super(AACF, self).__init__(config)
        train_matrix = self.dataset.train_matrix
        self.train_matrix = train_matrix
        self.users_num, self.items_num = train_matrix.shape

        self.factors_num = config["factors_num"]
        self.lr = config["lr"]
        self.g_reg = config["g_reg"]
        self.d_reg = config["d_reg"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.att_num = config["att_num"]
        self.att_top_k = config["att_top_k"]
        self.tau = config["tau"]

        self.user_pos_train = csr_to_user_dict(train_matrix)
        self._pre_data()
        self.pretrain_path = os.path.join(os.path.dirname(config["train_file"]), "_tmp_gan")
        self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def load_pretrain(self):
        if not os.path.exists(self.pretrain_path):
            os.makedirs(self.pretrain_path)
        file_path = os.path.join(self.pretrain_path, self.dataset.name+"_dns.npy")
        if os.path.isfile(file_path):
            print("loading pretraining parameters...")
            params = np.load(file_path, allow_pickle=True)
        else:
            print("pretraining model...")
            params = self._pretraining()
            np.save(file_path, params)
        return params

    def _pretraining(self):
        dnsbpr = DNSBPR(self.users_num, self.items_num, self.factors_num)
        dnsbpr.train_model(self.dataset.train_matrix, self.evaluator)
        return dnsbpr.parameters()

    def _pre_data(self):
        self.user_ids = list(self.user_pos_train.keys())
        all_users_list = []
        all_pos_items_list = []
        for user in self.user_ids:
            pos_items = self.user_pos_train[user]
            all_users_list.append(np.full_like(pos_items, user))
            all_pos_items_list.append(pos_items)

        self.all_users_list = np.concatenate(all_users_list)
        self.all_pos_items_list = np.concatenate(all_pos_items_list)

        rank = -np.arange(1, self.att_top_k + 1, dtype=np.float32)
        exp_logit = np.exp(rank/self.tau)
        self.att_top_prob = exp_logit/np.sum(exp_logit)
        assert len(self.att_top_prob) == self.att_top_k

    def _build_model(self):
        self.user_hd = tf.placeholder(tf.int32, shape=[None, ], name="user")
        self.pos_item_hd = tf.placeholder(tf.int32, shape=[None, ], name="pos_item")
        self.att_item_hd = tf.placeholder(tf.int32, shape=[None, None], name="att_item")

        pretrain_params = self.load_pretrain()

        self.generator = Generator(self.users_num, self.items_num, self.factors_num, params=pretrain_params)
        self.critic = Critic(self.users_num, self.items_num, self.factors_num)
        att_weight = self.generator.forward(self.user_hd, self.att_item_hd)
        critic_loss = self.critic.forward(self.user_hd, self.pos_item_hd, self.att_item_hd, att_weight)
        critic_loss = critic_loss + self.d_reg*tf.add_n(tf.get_collection("d_reg"))
        critic_opt = tf.train.GradientDescentOptimizer(self.lr)
        self.critic_update = critic_opt.minimize(critic_loss, var_list=self.critic.parameters())

        att_weight = self.generator.forward(self.user_hd, self.att_item_hd)
        gen_loss = -self.critic.forward(self.user_hd, self.pos_item_hd, self.att_item_hd, att_weight)
        gen_loss = gen_loss + self.g_reg*tf.add_n(tf.get_collection("g_reg"))
        gen_opt = tf.train.GradientDescentOptimizer(self.lr)
        self.gen_update = gen_opt.minimize(gen_loss, var_list=self.generator.parameters())

        # for evaluation
        users_embeddings, items_embeddings, items_biases = self.generator.parameters()
        user_embs = tf.nn.embedding_lookup(users_embeddings, self.user_hd)  # (b,d)
        self.batch_ratings = tf.matmul(user_embs, items_embeddings, transpose_b=True) + items_biases

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        result = self.evaluate_model()
        self.logger.info("Initialization:\t%s" % result)
        for epoch in range(self.epochs):
            all_att_items = self.get_att_items(self.att_num * 2, top_k=self.att_top_k)
            d_att_items, g_att_items = np.split(all_att_items, [self.att_num], axis=1)
            d_loader = self.get_train_data(d_att_items)
            g_loader = self.get_train_data(g_att_items)
            self.training_critic(d_loader)
            self.training_generator(g_loader)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    # @timer
    def training_critic(self, dataloader):
        for users, pos_items, att_items in dataloader:
            feed = {self.user_hd: users,
                    self.pos_item_hd: pos_items,
                    self.att_item_hd: att_items
                    }
            self.sess.run(self.critic_update, feed_dict=feed)

    # @timer
    def training_generator(self, dataloader):
        for users, pos_items, att_items in dataloader:
            feed = {self.user_hd: users,
                    self.pos_item_hd: pos_items,
                    self.att_item_hd: att_items
                    }
            self.sess.run(self.gen_update, feed_dict=feed)

    def get_train_data(self, neg_att_items):
        dataloader = DataIterator(self.all_users_list, self.all_pos_items_list, neg_att_items,
                                  batch_size=self.batch_size, shuffle=True)
        return dataloader

    # @timer
    def get_att_items(self, att_num, top_k=200):
        all_att_items = []
        user_iter = DataIterator(self.user_ids, batch_size=1024, shuffle=False, drop_last=False)
        for users in user_iter:
            ratings = self.sess.run(self.batch_ratings, feed_dict={self.user_hd: users}).squeeze()
            max_rating_items = arg_top_k(ratings, topk=top_k, n_threads=4)
            max_ratings = np.array([rating[max_idx] for max_idx, rating in zip(max_rating_items, ratings)])
            sample_probs = softmax(max_ratings, tau=self.tau)
            sample_size = [len(self.user_pos_train[u]) * att_num for u in users]
            att_items_idx = batch_randint_choice(top_k, sample_size, replace=True, p=sample_probs, thread_num=4)

            for user, att_items, att_idx in zip(users, max_rating_items, att_items_idx):
                pos_len = len(self.user_pos_train[user])
                att_items = att_items[att_idx]
                att_items = np.reshape(att_items, [pos_len, att_num])
                all_att_items.append(att_items)

        all_att_items = np.concatenate(all_att_items)
        return all_att_items

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users):
        feed = {self.user_hd: users}
        ratings = self.sess.run(self.batch_ratings, feed_dict=feed)
        return ratings
