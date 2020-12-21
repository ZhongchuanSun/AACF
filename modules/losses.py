import tensorflow as tf


def log_loss(yij, epsilon=1e-7, name="log_loss"):
    """ bpr loss
    :param yij:
    :param epsilon: A small increment to add to avoid taking a log of zero.
    :param name:
    :return:
    """
    with tf.name_scope(name):
        yij = tf.clip_by_value(yij, -80.0, 1e8)
        return -tf.log_sigmoid(yij)


def hinge_loss(yij, margin=1.0, name="hinge_loss"):
    return tf.nn.relu(margin - yij, name=name)


def square_loss(labels, logits, name="square_loss"):
    with tf.name_scope(name):
        return 0.5 * tf.square(labels-logits)


def cross_entropy_with_logits(labels, logits, name="cross_entropy_loss"):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name=name)
