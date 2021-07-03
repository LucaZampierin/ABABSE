#!/usr/bin/env python
# encoding: utf-8

# Neural network layers
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
import tensorflow.compat.v1 as tf

from utils import load_w2v


def bi_dynamic_rnn_abse(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    """
    Method adapted from Trusca et al. (2020). Bi-directional LSTM layer

    :param cell:
    :param inputs:
    :param n_hidden:
    :param length:
    :param max_len:
    :param scope_name:
    :param out_type:
    :return:
    """
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell(n_hidden),
        cell_bw=cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    if out_type == 'last':
        outputs_fw, outputs_bw = outputs
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
        outputs = tf.concat([outputs_fw, outputs_bw], 2)
    else:
        outputs_fw, outputs_bw = outputs
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
        outputs = tf.add(outputs_fw, outputs_bw)/2  # batch_size * max_len * n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, 2 * n_hidden]), index)  # batch_size * 2n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)  # batch_size * 2n_hidden
    return outputs


def stack_bi_dynamic_rnn_abse(cells_fw, cells_bw, inputs, n_hidden, n_layer, length, max_len, scope_name, out_type='last'):
    """
    Method adapted from Trusca et al. (2020). NOTE, this method is not directly sued in this project

    :param cells_fw:
    :param cells_bw:
    :param inputs:
    :param n_hidden:
    :param n_layer:
    :param length:
    :param max_len:
    :param scope_name:
    :param out_type:
    :return:
    """
    outputs, _, _ = tf.nn.stack_bidirectional_dynamic_rnn(
        cells_fw(n_hidden) * n_layer, cells_bw(n_hidden) * n_layer, inputs,
        sequence_length=length, dtype=tf.float32, scope=scope_name)
    if out_type == 'last':
        outputs_fw, outputs_bw = tf.split(2, 2, outputs)
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
        outputs = tf.concat([outputs_fw, outputs_bw], 2)
    else:
        outputs_fw, outputs_bw = outputs
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
        outputs = tf.add(outputs_fw, outputs_bw)/2  # batch_size * max_len * n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, 2 * n_hidden]), index)  # batch_size * 2n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)  # batch_size * 2n_hidden
    return outputs


def reduce_mean_with_len(inputs, length):
    """
    Method obtained from Trusca et al. (2020), original docstring below.

    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keepdims=False) / length
    return inputs


def decoder_layer(input_prob, input_aspect, n_hidden, n_class, n_aspects, random_base, l2_reg, sub_vocab, FLAGS, scope_name='1',  use_aspect=True):
    """
    Decoder structure of the autoencoder-like model taht reconstructs the sentence using the sentimenet embedding matrix

    :param input_prob:
    :param input_aspect:
    :param n_hidden:
    :param n_class:
    :param n_aspects:
    :param random_base:
    :param l2_reg:
    :param sub_vocab:
    :param FLAGS:
    :param scope_name:
    :param use_aspect:
    :return:
    """
    w = tf.get_variable(
        name='sentiment_embedding' + scope_name,
        shape=[n_class, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer= tf.keras.regularizers.L2(l2_reg),
        trainable=True
        )

    if use_aspect:
        w_aspect = tf.get_variable(
            name='aspect_w' + scope_name,
            shape=[n_aspects, n_hidden],
            # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
            initializer=tf.random_uniform_initializer(-random_base, random_base),
            regularizer=tf.keras.regularizers.L2(l2_reg),
            trainable=True
        )
    batch_size = tf.shape(input_prob)[0]
    if use_aspect:
        outputs = tf.matmul(input_prob, w) + tf.matmul(input_aspect, w_aspect)
    else:
        outputs = tf.matmul(input_prob, w)
    return outputs, w


def softmax_layer(inputs, n_hidden, random_base, drop_rate, l2_reg, n_class, scope_name='1'):
    """
    Method adapted from Trusca et al. (2020). Encodes the sentence representation into a three dimensional vector
    (sentiment classification) using a softmax function.

    :param inputs:
    :param n_hidden:
    :param random_base:
    :param drop_rate:
    :param l2_reg:
    :param n_class:
    :param scope_name:
    :return:
    """
    w = tf.get_variable(
        name='softmax_w' + scope_name,
        shape=[n_hidden, n_class],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.keras.regularizers.L2(l2_reg)
    )
    b = tf.get_variable(
        name='softmax_b' + scope_name,
        shape=[n_class],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_class))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.keras.regularizers.L2(l2_reg)
    )
    with tf.name_scope('softmax'):
        outputs = tf.nn.dropout(inputs, rate=drop_rate)
        predict = tf.matmul(outputs, w) + b
        predict = tf.nn.softmax(predict)
    return predict, w
