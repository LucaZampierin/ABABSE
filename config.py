#!/usr/bin/env python
# encoding: utf-8

# Configuration file.
#
# https://github.com/LucaZampierin/ABABSE
#
# Adapted to newest version of TensorFlow from Trusca, Wassenberg, Frasincar and Dekker (2020).
# https://github.com/mtrusca/HAABSA_PLUS_PLUS
#
# Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020). A Hybrid Approach for aspect-based sentiment analysis using
# deep contextual word embeddings and hierarchical attention. 20th International Conference on Web Engineering (ICWE 2020)
# (Vol.12128, pp. 365–380). Springer

import tensorflow as tf2
import sys
import numpy as np
import tensorflow.compat.v1 as tf
tf.executing_eagerly()
from utils import load_w2v
from nn_layer import reduce_mean_with_len

FLAGS = tf.app.flags.FLAGS
# General variables
tf.app.flags.DEFINE_integer("year",2016, "year data set (2015 or 2016)")
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 30, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct sentiment classes')
tf.app.flags.DEFINE_integer('n_aspect', 12, 'number of distint aspect categories')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 4, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.003, 'l2 regularization')
tf.app.flags.DEFINE_float('random_base', 0.1, 'initial random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 3, 'number of iterations used for training')
tf.app.flags.DEFINE_float('drop_rate1', 0.05, 'dropout rate')
tf.app.flags.DEFINE_float('drop_rate2', 0.05, 'dropout rate')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.app.flags.DEFINE_integer('n_layer', 20, 'number of stacked rnn')
tf.app.flags.DEFINE_string('is_r', '0', 'prob')
tf.app.flags.DEFINE_integer('max_target_len', 19, 'max target length')
tf.app.flags.DEFINE_integer('negative_samples', 10, "number of negative samples")
tf.app.flags.DEFINE_float('ortho_reg', 1, 'weight redundancy regularization term')
tf.app.flags.DEFINE_float('seed_reg', 5, 'weight seed regularization term')
tf.app.flags.DEFINE_integer('random_seed', 1234, 'seed used for random number generator')

# traindata, testdata and embeddings
tf.app.flags.DEFINE_string("train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'traindata'+str(FLAGS.year)+".txt", "train data path")
tf.app.flags.DEFINE_string("test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'testdata'+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("embedding_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'embedding'+str(FLAGS.year)+".txt", "pre-trained glove vectors file path")
tf.app.flags.DEFINE_string("remaining_test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

# hyper traindata, hyper testdata
tf.app.flags.DEFINE_string("hyper_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertraindata'+str(FLAGS.year)+".txt", "hyper train data path")
tf.app.flags.DEFINE_string("hyper_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevaldata'+str(FLAGS.year)+".txt", "hyper eval data path")

# external data sources
tf.app.flags.DEFINE_string("pretrain_file", "data/externalData/glove.42B."+str(FLAGS.embedding_dim)+"d.txt", "pre-trained glove vectors file path")
tf.app.flags.DEFINE_string("train_data", "data/externalData/restaurant_train_"+str(FLAGS.year)+".xml",
                    "train data path")
tf.app.flags.DEFINE_string("test_data", "data/externalData/restaurant_test_"+str(FLAGS.year)+".xml",
                    "test data path")

tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('prob_file', 'prob1.txt', 'prob')
tf.app.flags.DEFINE_string('saver_file', 'prob1.txt', 'prob')


def print_config():
    """
    Method obtained from Trusca et al. (2020). Prints the configuration.
    :return:
    """
    FLAGS(sys.argv)
    print('\nParameters:')
    for k, v in sorted(tf.app.flags.FLAGS.flag_values_dict().items()):
        print('{}={}'.format(k, v))


def ortho_reg(weight_matrix, ortho_reg_weight=FLAGS.ortho_reg):
    """
    Redundancy (orthogoanal) regularization for sentiment embedding matrix.

    :param weight_matrix: sentiment embedding matrix
    :param ortho_reg_weight: redundancy regularization weight
    :return: redundancy regularization
    """
    w_n = tf.divide(weight_matrix, tf.math.sqrt(tf.reduce_sum(tf.multiply(weight_matrix, weight_matrix), axis=1, keepdims=True)))
    reg = tf.sqrt(tf.reduce_sum(tf.math.square(tf.linalg.matmul(w_n, w_n, transpose_a=False, transpose_b=True) - tf.eye(tf.shape(w_n)[0]))))

    return ortho_reg_weight * reg

def seed_reg(weight_matrix, sub_vocab, seed_reg_weight=FLAGS.seed_reg):
    """
    Seed regularization for sentiment embedding matrix.

    :param weight_matrix: sentiment embedding matrix
    :param sub_vocab: sentiment seed words
    :param seed_reg_weight: seed regularization weight
    :return: seed regularization
    """
    word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
    word_embedding = tf.constant(w2v, name='word_embedding')

    if type(word_id_mapping) is str:
        word_to_id = load_word_id_mapping(word_id_mapping)
    else:
        word_to_id = word_id_mapping

    # Create prior matrix
    prior_matrix = tf.zeros([0, FLAGS.embedding_dim], dtype=tf.dtypes.float32)
    counter = 0
    for sub in sub_vocab:
        sentiment = tf.zeros([1, FLAGS.embedding_dim], dtype=tf.dtypes.float32)
        n_words = 0
        for word in sub:
            sentiment = tf.add(sentiment, tf.transpose(tf.nn.embedding_lookup(word_embedding, word_to_id[word])))
            n_words += 1
        prior_matrix = tf.concat([prior_matrix, sentiment/5], axis=0)
        counter += 1

    w_n = tf.divide(weight_matrix, tf.math.sqrt(tf.reduce_sum(tf.multiply(weight_matrix, weight_matrix), axis=1, keepdims=True)))
    prior_matrix = tf.divide(prior_matrix, tf.math.sqrt(tf.reduce_sum(tf.multiply(prior_matrix, prior_matrix), axis=1, keepdims=True)))
    sreg = FLAGS.n_class - tf.reduce_sum(tf.math.multiply(w_n, prior_matrix))
    return seed_reg_weight * sreg


def train_loss_func(z_s, r_s, sent_embedding, sub_vocab, neg_samples, seed_reg_weight, ortho_reg_weight, nsamples, neg_sen_len):
    """
    Computes the hinge loss for training

    :param z_s: sentence reprsentation
    :param r_s: sentence reconstruction
    :param sent_embedding: sentiment embedding matrix
    :param sub_vocab: sentiment seed words
    :param neg_samples: negative samples used for training
    :param seed_reg_weight: seed regularization weight
    :param ortho_reg_weight: orthogonal regularization weight
    :param nsamples: number of negative samples
    :param neg_sen_len: negative samples sentence length
    :return: trainining loss with regularizations
    """
    batch_size = tf.shape(r_s)[0]
    neg_samples = reduce_mean_with_len(neg_samples, neg_sen_len)
    pos = tf.divide(tf.reduce_sum(tf.math.multiply(z_s, r_s), axis=1, keepdims=True), tf.multiply(tf.math.sqrt(tf.reduce_sum(tf.square(z_s), axis=1, keepdims=True)), tf.math.sqrt(tf.reduce_sum(tf.square(r_s), axis=1, keepdims=True))))
    neg_samples = tf.reshape(neg_samples, [batch_size, nsamples, FLAGS.embedding_dim])
    r_s = tf.repeat(r_s, repeats=nsamples, axis=0)
    r_s = tf.reshape(r_s, shape=[batch_size, nsamples, FLAGS.embedding_dim])
    neg = tf.divide(tf.reduce_sum(tf.math.multiply(r_s, neg_samples), axis=-1), tf.multiply(tf.math.sqrt(tf.reduce_sum(tf.square(r_s), axis=-1)), tf.math.sqrt(tf.reduce_sum(tf.square(neg_samples), axis=-1))))
    neg = tf.reshape(neg, shape=[batch_size,nsamples])
    loss = (tf.math.maximum(0., 1. - pos + neg))

    loss = tf.reduce_mean(loss)

    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = loss  + sum(reg_loss) + seed_reg(weight_matrix=sent_embedding, sub_vocab=sub_vocab, seed_reg_weight=seed_reg_weight) + ortho_reg(weight_matrix=sent_embedding, ortho_reg_weight=ortho_reg_weight)
    return loss


def test_loss_func(z_s, r_s):
    """
    Computes the hinge loss for the testset

    :param z_s: test sentence representation
    :param r_s: test sentence reconstruction
    :return: test loss
    """
    pos = tf.divide(tf.reduce_sum(tf.math.multiply(z_s, r_s), axis=1, keepdims=True), tf.multiply(tf.math.sqrt(tf.reduce_sum(tf.square(z_s), axis=1, keepdims=True)), tf.math.sqrt(tf.reduce_sum(tf.square(r_s), axis=1, keepdims=True))))
    loss = (tf.maximum(0., tf.subtract(1., pos)))
    loss = tf.reduce_sum(loss)
    return loss


def loss_func(y, prob):
    """
    Method obtained from Trusca et al. (2020). Training loss for supervised models.
    NOTE. Not used in this project.

    :param y:
    :param prob:
    :return:
    """
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = - tf.reduce_mean(y * tf.log(prob)) + sum(reg_loss)
    return loss


def acc_func(y, prob):
    """
    Method obtained from Trusca et al. (2020). Computes the prediction accuracy.

    :param y: true labels
    :param prob: predicted labels
    :return: accuracy
    """
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1)) # MAKE SURE THE ORDER OF Y AND PROB IS THE SAME
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc_num, acc_prob


def summary_func(loss, acc, test_loss, test_acc, _dir, title, sess):
    """
    Method obtained from Trusca et al. (2020). Writes summary files.
    NOTE. Not used in this project

    :param loss:
    :param acc:
    :param test_loss:
    :param test_acc:
    :param _dir:
    :param title:
    :param sess:
    :return:
    """
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc = tf.summary.scalar('acc' + title, acc)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)
    train_summary_op = tf.summary.merge([summary_loss, summary_acc])
    validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(_dir + '/test')
    validate_summary_writer = tf.summary.FileWriter(_dir + '/validate')
    return train_summary_op, test_summary_op, validate_summary_op, \
        train_summary_writer, test_summary_writer, validate_summary_writer


def saver_func(_dir):
    """
    Method obtained from Trusca et al. (2020). File saver function
    :param _dir:
    :return:
    """
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)
    import os
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return saver
