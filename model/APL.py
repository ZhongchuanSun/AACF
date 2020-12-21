"""
Paper: APL: Adversarial Pairwise Learning for Recommender Systems
Author: Zhongchuan Sun, Bin Wu, Yunpeng Wu and Yangdong Ye
"""

from model.base import AbstractRecommender
from model.base import MatrixFactorization
import tensorflow as tf
from modules.losses import log_loss
from reckit import DataIterator
from reckit import timer
import numpy as np
from utils.tools import csr_to_user_dict
from .DNSBPR import DNSBPR
import os


def gumbel_softmax(logits, temperature=0.2):
    eps = 1e-20
    u = tf.random_uniform(tf.shape(logits), minval=0, maxval=1)
    gumbel_noise = -tf.log(-tf.log(u + eps) + eps)
    y = tf.log(logits + eps) + gumbel_noise
    return tf.nn.softmax(y / temperature)


class Generator(MatrixFactorization):
    def __init__(self, users_num, items_num, factors_num, params=None, name="generator"):
        super(Generator, self).__init__(users_num, items_num, factors_num, params=params, name=name)

    def forward(self, user, name="g_forward"):
        all_logits = self.all_logits(user, name=name)
        return all_logits

    def sampling_for_gen(self, user, p_aux):
        all_logit = self.all_logits(user)
        prob = tf.nn.softmax(all_logit)
        prob = (1-0.2)*prob + p_aux
        virtual_item = gumbel_softmax(prob, 0.2)
        self.reg_loss = self.user_l2loss(user) + self.item_l2loss()
        return virtual_item

    def sampling_for_critic(self, user):
        all_logit = self.all_logits(user)
        prob = tf.nn.softmax(all_logit/0.2)  # importance sampling
        virtual_item = gumbel_softmax(prob, 0.2)
        return virtual_item


class Critic(MatrixFactorization):
    def __init__(self, users_num, items_num, factors_num, name="discriminator"):
        super(Critic, self).__init__(users_num, items_num, factors_num, name=name)

    def get_loss(self, users, pos_items, neg_items, name="loss"):
        with tf.name_scope(name):
            yi_hat, reg_loss_1 = self._discrete_item_predict(users, pos_items)
            yj_hat, reg_loss_2 = self._virtual_item_predict(users, neg_items)
            loss = log_loss(yi_hat - yj_hat)
            self.reg_loss = reg_loss_1 + reg_loss_2
        return tf.reduce_mean(loss)

    def _discrete_item_predict(self, user, item, name="discrete_predict"):
        reg_loss = self.user_l2loss(user) + self.item_l2loss(item)
        return self.predict(user, item, name=name), reg_loss

    def _virtual_item_predict(self, user, item, name="virtual_predict"):
        with tf.name_scope(name):
            u_embedding = self.lookup_user_embedding(user)
            item_embeddings = self.lookup_item_embedding()
            item_bias = self.gather_item_bias()
            fake_one_hot = item
            i_embedding = tf.matmul(fake_one_hot, item_embeddings)
            i_bias = tf.reduce_sum(tf.multiply(fake_one_hot, item_bias), axis=1)
            logit = tf.reduce_sum(tf.multiply(u_embedding, i_embedding), 1) + i_bias
            reg_loss = tf.nn.l2_loss(u_embedding) + tf.nn.l2_loss(i_embedding) + tf.nn.l2_loss(i_bias)
        return logit, reg_loss


class APL(AbstractRecommender):
    def __init__(self, config):
        super(APL, self).__init__(config)
        train_matrix = self.dataset.train_matrix
        self.users_num, self.items_num = train_matrix.shape

        self.factors_num = config["factors_num"]
        self.lr = config["lr"]
        self.g_reg = config["g_reg"]
        self.d_reg = config["d_reg"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.user_pos_train = csr_to_user_dict(train_matrix)
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
        dnsbpr.train_model(self.dataset.train_matrix)
        return dnsbpr.parameters()

    def _build_model(self):
        self.user_holder = tf.placeholder(tf.int32)
        self.item_holder = tf.placeholder(tf.int32)
        self.g_aux_holder = tf.placeholder(tf.float32)

        pretrain_params = self.load_pretrain()

        self.generator = Generator(self.users_num, self.items_num, self.factors_num, params=pretrain_params)
        self.critic = Critic(self.users_num, self.items_num, self.factors_num)
        virtual_item = self.generator.sampling_for_critic(self.user_holder)
        critic_loss = self.critic.get_loss(self.user_holder, self.item_holder, virtual_item)
        critic_loss = critic_loss + self.d_reg*self.critic.reg_loss
        critic_opt = tf.train.GradientDescentOptimizer(self.lr)
        self.critic_update = critic_opt.minimize(critic_loss, var_list=self.critic.parameters())

        virtual_item = self.generator.sampling_for_gen(self.user_holder, self.g_aux_holder)
        gen_loss = -self.critic.get_loss(self.user_holder, self.item_holder, virtual_item)
        gen_loss = gen_loss + self.g_reg*self.generator.reg_loss
        gen_opt = tf.train.GradientDescentOptimizer(self.lr)
        self.gen_update = gen_opt.minimize(gen_loss, var_list=self.generator.parameters())

        # for evaluating
        users_embeddings, items_embeddings, items_biases = self.generator.parameters()
        batch_u_embeddings = tf.nn.embedding_lookup(users_embeddings, self.user_holder)
        self.batch_ratings = tf.matmul(batch_u_embeddings, items_embeddings, transpose_b=True) + items_biases

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        result = self.evaluate_model()
        self.logger.info("pretrain:\t%s" % result)

        dataloader = self.get_train_data()
        for epoch in range(self.epochs):
            self.training_critic(dataloader)
            self.training_generator(dataloader)
            result = self.evaluate_model()
            self.logger.info("%d:\t%s" % (epoch, result))

    # @timer
    def training_critic(self, dataloader):
        for input_user, pos_items in dataloader:
            feed = {self.user_holder: input_user,
                    self.item_holder: pos_items}
            self.sess.run(self.critic_update, feed_dict=feed)

    # @timer
    def training_generator(self, dataloader):
        for input_user, pos_items in dataloader:
            p_aux = np.zeros([len(pos_items), self.items_num])
            for idx, user in enumerate(input_user):
                p_aux[idx][self.user_pos_train[user]] = 0.2/len(self.user_pos_train[user])
            feed = {self.user_holder: input_user,
                    self.item_holder: pos_items,
                    self.g_aux_holder: p_aux}
            self.sess.run(self.gen_update, feed_dict=feed)

    def get_train_data(self):
        users_list = []
        items_list = []
        for user, items in self.user_pos_train.items():
            users_list.extend([user]*len(items))
            items_list.extend(items)

        dataloader = DataIterator(users_list, items_list, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users):
        feed = {self.user_holder: users}
        ratings = self.sess.run(self.batch_ratings, feed_dict=feed)
        return ratings
