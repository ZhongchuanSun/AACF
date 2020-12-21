import tensorflow as tf


class MatrixFactorization(object):
    def __init__(self, users_num, items_num, factors_num, params=None, name="MF"):
        self.users_num = users_num
        self.items_num = items_num
        self.factors_num = factors_num

        # initialize parameters
        with tf.name_scope(name):
            if params is not None:
                self.user_embeddings = tf.Variable(params[0], name="user_embedding")
                self.item_embeddings = tf.Variable(params[1], name="item_embedding")
                self.item_biases = tf.Variable(params[2], name="item_bias")
            else:
                self.user_embeddings = tf.Variable(tf.random_uniform([self.users_num, self.factors_num],
                                                                     minval=-0.05, maxval=0.05), name="user_embedding")
                self.item_embeddings = tf.Variable(tf.random_uniform([self.items_num, self.factors_num],
                                                                     minval=-0.05, maxval=0.05), name="item_embedding")
                self.item_biases = tf.Variable(tf.zeros([self.items_num]), name="item_bias")

    def parameters(self):
        return [self.user_embeddings, self.item_embeddings, self.item_biases]

    def predict(self, users, items, name="predict"):
        with tf.name_scope(name):
            user_embedding = self.lookup_user_embedding(users)
            item_embedding = self.lookup_item_embedding(items)
            item_bias = self.gather_item_bias(items)
            ratings = tf.reduce_sum(tf.multiply(user_embedding, item_embedding), axis=1) + item_bias
            return ratings

    def all_logits(self, users=None, name="all_logits"):
        """
        :param users:
        :param name:
        :return: If user is 'None', return all users ratings, otherwise return the ratings of given users
        """
        with tf.name_scope(name):
            if users is not None:
                user_embedding = self.lookup_user_embedding(users)
                all_rating = tf.matmul(user_embedding, self.item_embeddings, transpose_b=True) + self.item_biases
            else:
                all_rating = tf.matmul(self.user_embeddings, self.item_embeddings, transpose_b=True) + self.item_biases
        return all_rating

    def user_l2loss(self, users=None, name="user_l2loss"):
        with tf.name_scope(name):
            user_embedding = self.lookup_user_embedding(users)
            return tf.nn.l2_loss(user_embedding)

    def item_l2loss(self, items=None, name="item_l2loss"):
        with tf.name_scope(name):
            item_embedding = self.lookup_item_embedding(items)
            item_bias = self.gather_item_bias(items)
            return tf.nn.l2_loss(item_embedding) + tf.nn.l2_loss(item_bias)

    def lookup_user_embedding(self, user=None):
        if user is None:
            embedding = self.user_embeddings
        else:
            embedding = tf.nn.embedding_lookup(self.user_embeddings, user)
        return embedding

    def lookup_item_embedding(self, item=None):
        if item is None:
            embedding = self.item_embeddings
        else:
            embedding = tf.nn.embedding_lookup(self.item_embeddings, item)
        return embedding

    def gather_item_bias(self, item=None):
        if item is None:
            bias = self.item_biases
        else:
            bias = tf.gather(self.item_biases, item)
        return bias
