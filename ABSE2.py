#!/usr/bin/env python
# encoding: utf-8

# Aspect-Based Sentiment Extraction 2 (ABSE2).
#
# https://github.com/LucaZampierin/ABSE
#
# Adapted from Trusca, Wassenberg, Frasincar and Dekker (2020). Changes have been made to adapt the methods
# to the current project and to adapt the scripts to TensorFlow 2.5.
# https://github.com/mtrusca/HAABSA_PLUS_PLUS
#
# Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020). A Hybrid Approach for aspect-based sentiment analysis using
# deep contextual word embeddings and hierarchical attention. 20th International Conference on Web Engineering (ICWE 2020)
# (Vol.12128, pp. 365–380). Springer

import os, sys

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
import numpy as np

from nn_layer import softmax_layer, reduce_mean_with_len, decoder_layer, bi_dynamic_rnn_abse
from att_layer import bilinear_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_twitter, load_word_id_mapping

sys.path.append(os.getcwd())
tf.compat.v1.disable_eager_execution()


def abse2(input, sen_len, target, sen_len_tr, aspects, n_aspects, drop_rate1, drop_rate2, sub_vocab, l2, _id='all'):
    """
    Structure of the Aspect-Based Sentiment Extraction 2 (ABSE2) attentional neural network.
    Adapts the strutures in Trusca et al. (2020) to the current project.

    :param input:
    :param sen_len:
    :param target:
    :param sen_len_tr:
    :param aspects:
    :param n_aspects:
    :param drop_rate1:
    :param drop_rate2:
    :param sub_vocab:
    :param l2:
    :param _id:
    :return:
    """
    print('I am ABSE2.')
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell
    input = tf.nn.dropout(input, rate=drop_rate1)
    hiddens = bi_dynamic_rnn_abse(cell, input, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'l' + _id, 'all')
    target = bi_dynamic_rnn_abse(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_target_len, 'target' + _id, 'all')
    pool_t = reduce_mean_with_len(target, sen_len_tr)

    # Attention layer.
    att = bilinear_attention_layer(hiddens, pool_t, sen_len, FLAGS.n_hidden, l2, FLAGS.random_base, 'tl')

    # Sentence representation.
    z_s = tf.matmul(att, input)
    z_s = tf.squeeze(z_s, axis=1)

    # Autoencoder-like dimensionalty reduction. Senitment classification.
    prob, soft_weight = softmax_layer(z_s, FLAGS.n_hidden, FLAGS.random_base, drop_rate2, l2, FLAGS.n_class)

    # Sentence reconstruction.
    r_s, sent_embedding = decoder_layer(prob, aspects, FLAGS.n_hidden, FLAGS.n_class, n_aspects, FLAGS.random_base, l2,
                                        sub_vocab, FLAGS, use_aspect=True)
    return prob, r_s, z_s, att, sent_embedding


def main(train_path, test_path, test_size, sub_vocab, learning_rate=FLAGS.learning_rate, drop_rate=FLAGS.drop_rate1,
         b1=0.99, b2=0.99, l2=FLAGS.l2_reg, seed_reg=FLAGS.seed_reg, ortho_reg=FLAGS.ortho_reg, batchsize=FLAGS.batch_size,
         nsamples=FLAGS.negative_samples):
    """
    Runs the ABSE2 method. Method adapted from Trusca et al. (2020) to the current project.

    :param train_path: path for train data
    :param test_path: path for test data
    :param test_size: size of test set
    :param sub_vocab: seed vocabulary for seed regularization
    :param learning_rate: learning rate used for backpropagation, defaults to 0.005
    :param drop_rate: dropout rate, defaults to 0.05
    :param b1: beta1 hyperparameter of Adam optimizer, defaults to 0.99
    :param b2: beta2 hyperparameter of Adam optimizer, defaults to 0.99
    :param l2: L2 regularization weight, defaults to 0.003
    :param seed_reg: seed regularization weight, defaults to 5
    :param ortho_reg: redundancy (orthogonal) regularization weight, defaults to 1
    :param batchsize: number of training observations in each batch, defaults to 30
    :param nsamples: number of negative samples for training, defaults to 10
    :return:
    """
    print_config()
    with tf.device('/gpu:1'):
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        word_embedding = tf.constant(w2v, dtype=np.float32, name='word_embedding')

        drop_rate1 = tf.placeholder(tf.float32)
        drop_rate2 = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            aspect = tf.placeholder(tf.float32, [None, FLAGS.n_aspect])
            sen_len = tf.placeholder(tf.int32, None)

            ns_words = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            neg_sen_len = tf.placeholder(tf.int32, None)

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len = tf.placeholder(tf.int32, [None])

        inputs = tf.nn.embedding_lookup(word_embedding, x)
        target = tf.nn.embedding_lookup(word_embedding, target_words)
        neg_samples = tf.nn.embedding_lookup(word_embedding, ns_words)

        att= None
        prob, r_s, z_s, att, sent_embedding = abse2(inputs, sen_len, target, tar_len, aspect, FLAGS.n_aspect,
                                                    drop_rate1, drop_rate2, sub_vocab, l2, 'all')

        loss = train_loss_func(z_s, r_s, sent_embedding, sub_vocab, neg_samples, seed_reg, ortho_reg, nsamples, neg_sen_len )
        t_loss = test_loss_func(z_s, r_s)

        acc_num, acc_prob = acc_func(y, prob) # acc_num is the total number of correct predictions, acc_prob is the percentage accuracy
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=b1, beta2=b2, epsilon=1e-08).minimize(loss,
                                                                                                        global_step=global_step)

        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

        title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
            FLAGS.drop_rate1,
            FLAGS.drop_rate2,
            FLAGS.batch_size,
            learning_rate,
            FLAGS.l2_reg,
            FLAGS.max_sentence_len,
            FLAGS.embedding_dim,
            FLAGS.n_hidden,
            FLAGS.n_class
        )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        # saver = saver_func(save_dir)

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, '/-')

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        tr_x, tr_sen_len, tr_target_word, tr_tar_len, tr_y, tr_aspect, _, tr_x_neg, tr_neg_sen_len = load_inputs_twitter(
            train_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'IAN',
            is_r,
            FLAGS.max_target_len,

        )
        te_x, te_sen_len, te_target_word, te_tar_len, te_y, te_aspect, all_y, te_x_neg, te_neg_sen_len = load_inputs_twitter(
            test_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'IAN',
            is_r,
            FLAGS.max_target_len
        )

        def get_batch_data(x_f, sen_len_f, yi, target, tl, aspecti, train_x_neg, sen_len_neg, batch_size,
                           negative_samples, dr1, dr2, is_shuffle=True):
            """
            Method adapted from Trusca et al. (2020). Obtain a batch of data.

            :param x_f:
            :param sen_len_f:
            :param yi:
            :param target:
            :param tl:
            :param aspecti:
            :param train_x_neg:
            :param sen_len_neg:
            :param batch_size:
            :param negative_samples:
            :param dr1:
            :param dr2:
            :param is_shuffle:
            :return:
            """
            for index, neg_index in batch_index(len(yi), batch_size, negative_samples, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    sen_len: sen_len_f[index],
                    y: yi[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    aspect: aspecti[index],
                    ns_words: train_x_neg[neg_index],
                    neg_sen_len: sen_len_neg[neg_index],
                    drop_rate1: dr1,
                    drop_rate2: dr2,
                }
                yield feed_dict, len(index)

        max_acc, max_train_acc, min_cost = 0., 0., 1000000.
        best_epoch = 0
        max_ty, max_py = None, None
        max_prob = None
        best_sent_emb = None
        step = None
        for i in range(FLAGS.n_iter):
            trainacc, traincnt, cost_train = 0., 0, 0.
            for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_y, tr_target_word, tr_tar_len, tr_aspect, tr_x_neg,
                                                  tr_neg_sen_len, batchsize, nsamples, drop_rate, drop_rate):
                loss_, _, step, summary, _trainacc = sess.run([loss, optimizer, global_step, train_summary_op, acc_num], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                trainacc += _trainacc
                traincnt += numtrain
                cost_train += loss_
            acc, cost, cnt = 0., 0., 0
            at, ty, py = [], [], []
            p = []
            # Test model
            for test, num in get_batch_data(te_x, te_sen_len, te_y, te_target_word, te_tar_len, te_aspect, te_x_neg, te_neg_sen_len,
                                            1000, 0, 0, 0, False):
                _loss, _acc, _ty, _py, _p, _att, sent_emb = sess.run(
                    [t_loss, acc_num, true_y, pred_y, prob, att, sent_embedding], feed_dict=test)
                ty = np.asarray(_ty)
                py = np.asarray(_py)
                p = np.asarray(_p)
                at = np.asarray(_att)
                sent_emb = np.asarray(sent_emb)
                acc += _acc
                cost += _loss * num
                cnt += num
            print('all samples={}, correct prediction={}'.format(cnt, acc))
            trainacc = trainacc / traincnt
            acc = acc / cnt
            totalacc = acc
            cost = cost / cnt
            print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, test acc={:.6f}, test_loss={:.6f}'.format(i, cost_train, trainacc, acc, cost))
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)

            if cost < min_cost and i != 0:
                min_cost = cost
                max_acc = acc
                max_train_acc = trainacc
                max_att = at
                max_ty = ty
                max_py = py
                max_prob = p
                best_epoch = i
                best_sent_emb = sent_emb

        P = precision_score(max_ty, max_py, average=None)
        R = recall_score(max_ty, max_py, average=None)
        F1 = f1_score(max_ty, max_py, average=None)
        print('P:', P, 'avg=', sum(P) / FLAGS.n_class)
        print('R:', R, 'avg=', sum(R) / FLAGS.n_class)
        print('F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)

        # Obtain the precision, recall, and F1 score for each class
        per_class_scores = np.concatenate([P.reshape(-1,3), R.reshape(-1,3), F1.reshape(-1,3)], axis=0)

        fp = open(FLAGS.prob_file, 'w')
        for item in max_prob:
            fp.write(' '.join([str(it) for it in item]) + '\n')
        fp = open(FLAGS.prob_file + '_att', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_att):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

        print('Optimization Finished! Max acc={}'.format(max_acc))

        print('Learning_rate={}, iter_num={}, batch_size={}, dropout_rate={}, l2={}, seed_reg={}, ortho_reg={}, negative_samples={}'.format(
            learning_rate,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.drop_rate1,
            FLAGS.l2_reg,
            FLAGS.seed_reg,
            FLAGS.ortho_reg,
            FLAGS.negative_samples
        ))
        return min_cost, best_epoch, max_acc, np.where(np.subtract(max_py, max_ty) == 0, 0, 1), max_att.tolist(), \
               all_y.tolist(), sum(P) / FLAGS.n_class, sum(R) / FLAGS.n_class, sum(F1) / FLAGS.n_class,\
               max_train_acc, best_sent_emb, per_class_scores


if __name__ == '__main__':
    tf.app.run()