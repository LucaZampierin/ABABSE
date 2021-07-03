#!/usr/bin/env python
# encoding: utf-8

# Attention layers.
#
# https://github.com/LucaZampierin/ABABSE
#
# Adapted from Trusca, Wassenberg, Frasincar and Dekker (2020). Changes have been made to adapt the methods
# to the current project and to adapt the scripts to TensorFlow 2.5.
# https://github.com/mtrusca/HAABSA_PLUS_PLUS
#
# Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020). A Hybrid Approach for aspect-based sentiment analysis using
# deep contextual word embeddings and hierarchical attention. 20th International Conference on Web Engineering (ICWE 2020)
# (Vol.12128, pp. 365–380). Springer

import numpy as np
import tensorflow as tf


def softmax_with_len(inputs, length, max_len):
    """
    Method obtained from Trusca et al. (2020). Computes softmax probabilities

    :param inputs: attention scores for each word in the sentence
    :param length: length of each sentence
    :param max_len:
    :return: attention weights
    """
    inputs = tf.cast(inputs, tf.float32)
    inputs = tf.exp(inputs)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.compat.v1.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
    return inputs / _sum


def bilinear_attention_layer(inputs, attend, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    Method obtained from Trusca et al. (2020), original docstring below, and adapted to TensorFlow 2.5.

    :param inputs: batch * max_len * n_hidden
    :param attend: batch * n_hidden
    :param length:
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id:
    :return:
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.compat.v1.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer = tf.keras.regularizers.L2(l2_reg),
        trainable=True
    )
    b = tf.compat.v1.get_variable(
        name='att_b_' + str(layer_id),
        shape=[1],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer = tf.keras.regularizers.L2(l2_reg),
        trainable=True
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_len, n_hidden])
    attend = tf.expand_dims(attend, 2)   # adds a dimension 1 at axis=2
    tmp = tf.matmul(tmp, attend)
    tmp = tmp + b
    tmp = tf.tanh(tmp)
    tmp = tf.reshape(tmp, shape=[batch_size, -1, max_len])
    return softmax_with_len(tmp, length, max_len)
