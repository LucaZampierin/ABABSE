# Main file to run all the models
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

import tensorflow as tf
import ABABSE1
import ABABSE2
import lcrModelU
from loadData import *

# import parameter configuration and data paths
from config import *
from utils import load_w2v, index_to_word

# import modules
import numpy as np
import sys
import matplotlib.pyplot as plt


# main function
def main(_):
    """
    Main method that runs the models.

    :param _:
    :return:
    """
    loadData = False    # Only needed when running the code for the first time for each dataset
    runABABSE1 = False    # Run ABABSE1 on the given dataset
    runABABSE2 = True    # Run ABABSE2 on the given dataset
    runLCRROTU = False   # Run Uns-LCR-Rot on the given dataset

    # Retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)
    test = FLAGS.test_path

    # ABABSE1 model
    if runABABSE1 == True:   #CHANGE VALUES THAT ARE RETURNED
        vocab_pos = {'amazing': 0, 'great': 1, 'nice': 2, 'impeccable': 3, 'excellent': 4}
        vocab_neg = {'rude': 0, 'bad': 1, 'terrible': 2, 'awful': 3, 'horrible': 4}
        vocab_neu = {'mediocre': 0, 'ordinary': 1, 'decent': 2, 'average': 3, 'ok': 4}
        sub_vocab = [vocab_pos, vocab_neu, vocab_neg]

        # Obtain average results over ten different runs to reduce effect of random fluctuations.
        number_runs = 10
        for i in range(number_runs):
            min_loss, best_epoch, acc, pred1, att, true, pr, rec, f1, tr_acc, sent_emb, per_class_scores = \
                ABABSE1.main(FLAGS.train_path, test, test_size, sub_vocab)

            acc = np.reshape(acc, (-1, 1))
            pr = np.reshape(pr, (-1, 1))
            rec = np.reshape(rec, (-1, 1))
            f1 = np.reshape(f1, (-1, 1))
            tr_acc = np.reshape(tr_acc, (-1, 1))
            if i == 0:
                accuracy = acc
                precision = pr
                recall = rec
                F1 = f1
                train_accuracy = tr_acc
                scores = per_class_scores
            else:
                accuracy += acc
                precision += pr
                recall += rec
                F1 += f1
                train_accuracy += tr_acc
                scores += per_class_scores
            tf.reset_default_graph()

        average_acc = accuracy / number_runs
        average_pr = precision / number_runs
        average_rec = recall / number_runs
        average_f1 = F1 / number_runs
        average_tr_acc = train_accuracy / number_runs
        scores = scores / number_runs

        print(average_tr_acc)
        print(average_acc)
        print(average_pr)
        print(average_rec)
        print(average_f1)
        print(scores)

        # Find ten nearest-neighbors to the learned sentiment embeddings
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        similarity = np.matmul(w2v, np.transpose(sent_emb))
        similarity = np.divide(similarity, np.linalg.norm(sent_emb, axis=1))
        similarity = np.divide(similarity, np.reshape(np.linalg.norm(w2v, axis=1), (-1, 1)))
        similar_indeces = np.argsort(similarity, axis=0)
        print('positive words')
        for ind in range(11):
            most_similar_positive = similar_indeces[-(ind + 1), 0]
            print(index_to_word(FLAGS.embedding_path, most_similar_positive))
        print('neutral words')
        for ind in range(11):
            most_similar_neutral = similar_indeces[-(ind + 1), 1]
            print(index_to_word(FLAGS.embedding_path, most_similar_neutral))
        print('negative words')
        for ind in range(11):
            most_similar_negative = similar_indeces[-(ind + 1), 2]
            print(index_to_word(FLAGS.embedding_path, most_similar_negative))

    # ABABSE2 model
    if runABABSE2 == True:

        vocab_pos = {'amazing': 0, 'great': 1, 'nice': 2, 'impeccable': 3, 'excellent': 4}
        vocab_neg = {'rude': 0, 'bad': 1, 'terrible': 2, 'awful': 3, 'horrible': 4}
        vocab_neu = {'mediocre': 0, 'ordinary': 1, 'decent': 2, 'average': 3, 'ok': 4}
        sub_vocab = [ vocab_pos, vocab_neu, vocab_neg]

        # Obtain average results over ten different runs to reduce effect of random fluctuations.
        number_runs = 10
        for i in range(number_runs):
            min_loss, best_epoch, acc, pred1, att, true, pr, rec, f1, tr_acc, sent_emb, per_class_scores = \
                ABABSE2.main(FLAGS.train_path, test, test_size, sub_vocab)

            acc= np.reshape(acc, (-1, 1))
            pr = np.reshape(pr, (-1, 1))
            rec = np.reshape(rec, (-1, 1))
            f1 = np.reshape(f1, (-1, 1))
            tr_acc = np.reshape(tr_acc, (-1, 1))
            if i == 0:
                accuracy = acc
                precision = pr
                recall = rec
                F1 = f1
                train_accuracy = tr_acc
                scores = per_class_scores
            else:
                accuracy += acc
                precision += pr
                recall += rec
                F1 += f1
                train_accuracy += tr_acc
                scores += per_class_scores
            tf.reset_default_graph()

        average_acc = accuracy/number_runs
        average_pr = precision/number_runs
        average_rec = recall/number_runs
        average_f1 = F1/number_runs
        average_tr_acc = train_accuracy/number_runs
        scores = scores/number_runs

        print(average_tr_acc)
        print(average_acc)
        print(average_pr)
        print(average_rec)
        print(average_f1)
        print(scores)

        # Find ten nearest-neighbors to the learned sentiment embeddings
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        similarity = np.matmul(w2v, np.transpose(sent_emb))
        similarity = np.divide(similarity, np.linalg.norm(sent_emb,axis = 1))
        similarity = np.divide(similarity, np.reshape(np.linalg.norm(w2v, axis=1), (-1,1)))
        similar_indeces = np.argsort(similarity, axis=0)
        print('positive words')
        for ind in range(11):
            most_similar_positive = similar_indeces[-(ind+1), 0]
            print(index_to_word(FLAGS.embedding_path, most_similar_positive))
        print('neutral words')
        for ind in range(11):
            most_similar_neutral = similar_indeces[-(ind+1), 1]
            print(index_to_word(FLAGS.embedding_path, most_similar_neutral))
        print('negative words')
        for ind in range(11):
            most_similar_negative = similar_indeces[-(ind + 1), 2]
            print(index_to_word(FLAGS.embedding_path, most_similar_negative))

    # Uns-LCR-Rot model
    if runLCRROTU == True:
        vocab_pos = {'amazing': 0, 'great': 1, 'nice': 2, 'impeccable': 3, 'excellent': 4}
        vocab_neg = {'rude': 0, 'bad': 1, 'terrible': 2, 'awful': 3, 'horrible': 4}
        vocab_neu = {'mediocre': 0, 'ordinary': 1, 'decent': 2, 'average': 3, 'ok': 4}
        sub_vocab = [ vocab_pos, vocab_neu, vocab_neg]

        # Obtain average results over ten different runs to reduce effect of random fluctuations.
        number_runs = 10
        for i in range(number_runs):
            min_loss, best_epoch, acc, pred1, fw1, bw1, tl1, tr1, true, pr, rec, f1, tr_acc, sent_emb, per_class_scores = \
                lcrModelU.main(FLAGS.train_path, test, test_size, sub_vocab)

            acc= np.reshape(acc, (-1, 1))
            pr = np.reshape(pr, (-1, 1))
            rec = np.reshape(rec, (-1, 1))
            f1 = np.reshape(f1, (-1, 1))
            tr_acc = np.reshape(tr_acc, (-1, 1))
            if i == 0:
                accuracy = acc
                precision = pr
                recall = rec
                F1 = f1
                train_accuracy = tr_acc
                scores = per_class_scores
            else:
                accuracy += acc
                precision += pr
                recall += rec
                F1 += f1
                train_accuracy += tr_acc
                scores += per_class_scores
            tf.reset_default_graph()

        average_acc = accuracy / number_runs
        average_pr = precision / number_runs
        average_rec = recall / number_runs
        average_f1 = F1 / number_runs
        average_tr_acc = train_accuracy / number_runs
        scores = scores / number_runs

        print(average_tr_acc)
        print(average_acc)
        print(average_pr)
        print(average_rec)
        print(average_f1)
        print(scores)

        # Find ten nearest-neighbors to the learned sentiment embeddings
        word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        similarity = np.matmul(w2v, np.transpose(sent_emb))
        similarity = np.divide(similarity, np.linalg.norm(sent_emb,axis = 1))
        similarity = np.divide(similarity, np.reshape(np.linalg.norm(w2v, axis=1), (-1,1)))
        similar_indeces = np.argsort(similarity, axis=0)
        print('positive words')
        for ind in range(11):
            most_similar_positive = similar_indeces[-(ind+1), 0]
            print(index_to_word(FLAGS.embedding_path, most_similar_positive))
        print('neutral words')
        for ind in range(11):
            most_similar_neutral = similar_indeces[-(ind+1), 1]
            print(index_to_word(FLAGS.embedding_path, most_similar_neutral))
        print('negative words')
        for ind in range(11):
            most_similar_negative = similar_indeces[-(ind + 1), 2]
            print(index_to_word(FLAGS.embedding_path, most_similar_negative))

    print('Finished program succesfully')


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
