import tensorflow as tf


def inner_product(a, b, name="inner_product"):
    with tf.name_scope(name=name):
        return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def scaled_inner_product(a, b, name="scaled_inner_product"):
    raise NotImplementedError("NotImplementedError")


dot_product = inner_product
scaled_dot_product = scaled_inner_product


def euclidean_distance(a, b, name="euclidean_distance"):
    return tf.norm(a - b, ord='euclidean', axis=-1, name=name)
