# Main file to run cross-validation. NOTE: not used in this project
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

import tensorflow as tf
import ABSE1
import ABSE2
import lcrModelU
from loadData import *

# import parameter configuration and data paths
from config import *

# import modules
import numpy as np
import sys


# main function
def main(_):
    """
    Main method that runs cross validation for all the models.

    :param _:
    :return:
    """
    loadData = False    # Only needed when running the code for the first time for each dataset
    runABSE1 = False    # Run ABSE1 on the given dataset
    runABSE2 = True    # Run ABSE2 on the given dataset
    runLCRROTU = False   # Run Uns-LCR-Rot on the given dataset

    train_path = "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_train_'
    val_path = "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_val_'

    # Number of k-fold cross validations
    split_size = 10

    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadCrossValidation(FLAGS, split_size,
                                                                                             loadData)

    # ABSE1 model
    if runABSE1 == True:  # CHANGE VALUES THAT ARE RETURNED
        acc = []
        pr = []
        rec = []
        f1 = []
        tr_acc = []

        vocab_pos = {'amazing': 0, 'great': 1, 'nice': 2, 'impeccable': 3, 'excellent': 4}
        vocab_neg = {'rude': 0, 'bad': 1, 'terrible': 2, 'awful': 3, 'horrible': 4}
        vocab_neu = {'mediocre': 0, 'ordinary': 1, 'decent': 2, 'average': 3, 'ok': 4}
        sub_vocab = [vocab_pos, vocab_neu, vocab_neg]

        # k-fold cross validation
        for i in range(split_size):
            # Obtain average results over ten different runs to reduce effect of random fluctuations.
            number_runs = 10
            for i in range(number_runs):
                _, _, acc_, _, _, _, pr_, rec_, f1_, tr_acc_, _, _ = \
                    ABSE1.main(train_path, val_path, test_size, sub_vocab)

                acc_ = np.reshape(acc_, (-1, 1))
                pr_ = np.reshape(pr_, (-1, 1))
                rec_ = np.reshape(rec_, (-1, 1))
                f1_ = np.reshape(f1_, (-1, 1))
                tr_acc_ = np.reshape(tr_acc_, (-1, 1))
                if i == 0:
                    accuracy = acc_
                    precision = pr_
                    recall = rec_
                    F1 = f1_
                    train_accuracy = tr_acc_
                else:
                    accuracy += acc_
                    precision += pr_
                    recall += rec_
                    F1 += f1_
                    train_accuracy += tr_acc_
                tf.reset_default_graph()

            average_acc = accuracy / number_runs
            average_pr = precision / number_runs
            average_rec = recall / number_runs
            average_f1 = F1 / number_runs
            average_tr_acc = train_accuracy / number_runs

            print(average_tr_acc)
            print(average_acc)
            print(average_pr)
            print(average_rec)
            print(average_f1)

            acc.append(average_acc)
            pr.append(average_pr)
            rec.append(average_rec)
            f1.append(average_f1)
            tr_acc.append(average_tr_acc)

        with open("cross_results_" + str(FLAGS.year) + "/ABSE1_" + str(FLAGS.year) + '.txt',
                  'w') as result:
            result.write(str(acc))
            result.write('Test Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
            result.write('Train Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(tr_acc)), np.std(np.asarray(tr_acc))))
            result.write('Test Precision: {}, St Dev:{} /n'.format(np.mean(np.asarray(pr)), np.std(np.asarray(pr))))
            result.write('Test Recall: {}, St Dev:{} /n'.format(np.mean(np.asarray(rec)), np.std(np.asarray(rec))))
            result.write('Test F1 score: {}, St Dev:{} /n'.format(np.mean(np.asarray(f1)), np.std(np.asarray(f1))))
            print(str(split_size) + '-fold cross validation results')
            print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))

    # ABSE2 model
    if runABSE2 == True:  # CHANGE VALUES THAT ARE RETURNED
        acc = []
        pr = []
        rec = []
        f1 = []
        tr_acc = []

        vocab_pos = {'amazing': 0, 'great': 1, 'nice': 2, 'impeccable': 3, 'excellent': 4}
        vocab_neg = {'rude': 0, 'bad': 1, 'terrible': 2, 'awful': 3, 'horrible': 4}
        vocab_neu = {'mediocre': 0, 'ordinary': 1, 'decent': 2, 'average': 3, 'ok': 4}
        sub_vocab = [vocab_pos, vocab_neu, vocab_neg]

        # k-fold cross validation
        for i in range(split_size):
            # Obtain average results over ten different runs to reduce effect of random fluctuations.
            number_runs = 10
            for i in range(number_runs):
                _, _, acc_, _, _, _, pr_, rec_, f1_, tr_acc_, _, _ = \
                    ABSE2.main(train_path, val_path, test_size, sub_vocab)

                acc_ = np.reshape(acc_, (-1, 1))
                pr_ = np.reshape(pr_, (-1, 1))
                rec_ = np.reshape(rec_, (-1, 1))
                f1_ = np.reshape(f1_, (-1, 1))
                tr_acc_ = np.reshape(tr_acc_, (-1, 1))
                if i == 0:
                    accuracy = acc_
                    precision = pr_
                    recall = rec_
                    F1 = f1_
                    train_accuracy = tr_acc_
                else:
                    accuracy += acc_
                    precision += pr_
                    recall += rec_
                    F1 += f1_
                    train_accuracy += tr_acc_
                tf.reset_default_graph()

            average_acc = accuracy / number_runs
            average_pr = precision / number_runs
            average_rec = recall / number_runs
            average_f1 = F1 / number_runs
            average_tr_acc = train_accuracy / number_runs

            print(average_tr_acc)
            print(average_acc)
            print(average_pr)
            print(average_rec)
            print(average_f1)

            acc.append(average_acc)
            pr.append(average_pr)
            rec.append(average_rec)
            f1.append(average_f1)
            tr_acc.append(average_tr_acc)

        with open("cross_results_" + str(FLAGS.year) + "/ABSE2_" + str(FLAGS.year) + '.txt',
                  'w') as result:
            result.write(str(acc))
            result.write(
                'Test Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
            result.write(
                'Train Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(tr_acc)), np.std(np.asarray(tr_acc))))
            result.write('Test Precision: {}, St Dev:{} /n'.format(np.mean(np.asarray(pr)), np.std(np.asarray(pr))))
            result.write('Test Recall: {}, St Dev:{} /n'.format(np.mean(np.asarray(rec)), np.std(np.asarray(rec))))
            result.write('Test F1 score: {}, St Dev:{} /n'.format(np.mean(np.asarray(f1)), np.std(np.asarray(f1))))
            print(str(split_size) + '-fold cross validation results')
            print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))

    # Uns-LCR-Rot model
    if runLCRROTU == True:  # CHANGE VALUES THAT ARE RETURNED
        acc = []
        pr = []
        rec = []
        f1 = []
        tr_acc = []

        vocab_pos = {'amazing': 0, 'great': 1, 'nice': 2, 'impeccable': 3, 'excellent': 4}
        vocab_neg = {'rude': 0, 'bad': 1, 'terrible': 2, 'awful': 3, 'horrible': 4}
        vocab_neu = {'mediocre': 0, 'ordinary': 1, 'decent': 2, 'average': 3, 'ok': 4}
        sub_vocab = [vocab_pos, vocab_neu, vocab_neg]

        # k-fold cross validation
        for i in range(split_size):
            # Obtain average results over ten different runs to reduce effect of random fluctuations.
            number_runs = 10
            for i in range(number_runs):
                _, _, acc_, _, _, _, _, _, _, pr_, rec_, f1_, tr_acc_, _, _ = \
                    lcrModelU.main(train_path, val_path, test_size, sub_vocab)

                acc_ = np.reshape(acc_, (-1, 1))
                pr_ = np.reshape(pr_, (-1, 1))
                rec_ = np.reshape(rec_, (-1, 1))
                f1_ = np.reshape(f1_, (-1, 1))
                tr_acc_ = np.reshape(tr_acc_, (-1, 1))
                if i == 0:
                    accuracy = acc_
                    precision = pr_
                    recall = rec_
                    F1 = f1_
                    train_accuracy = tr_acc_
                else:
                    accuracy += acc_
                    precision += pr_
                    recall += rec_
                    F1 += f1_
                    train_accuracy += tr_acc_
                tf.reset_default_graph()

            average_acc = accuracy / number_runs
            average_pr = precision / number_runs
            average_rec = recall / number_runs
            average_f1 = F1 / number_runs
            average_tr_acc = train_accuracy / number_runs

            print(average_tr_acc)
            print(average_acc)
            print(average_pr)
            print(average_rec)
            print(average_f1)

            acc.append(average_acc)
            pr.append(average_pr)
            rec.append(average_rec)
            f1.append(average_f1)
            tr_acc.append(average_tr_acc)

        with open("cross_results_" + str(FLAGS.year) + "/Uns_LCR_Rot_" + str(FLAGS.year) + '.txt',
                  'w') as result:
            result.write(str(acc))
            result.write(
                'Test Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))
            result.write(
                'Train Accuracy: {}, St Dev:{} /n'.format(np.mean(np.asarray(tr_acc)), np.std(np.asarray(tr_acc))))
            result.write('Test Precision: {}, St Dev:{} /n'.format(np.mean(np.asarray(pr)), np.std(np.asarray(pr))))
            result.write('Test Recall: {}, St Dev:{} /n'.format(np.mean(np.asarray(rec)), np.std(np.asarray(rec))))
            result.write('Test F1 score: {}, St Dev:{} /n'.format(np.mean(np.asarray(f1)), np.std(np.asarray(f1))))
            print(str(split_size) + '-fold cross validation results')
            print('Accuracy: {}, St Dev:{}'.format(np.mean(np.asarray(acc)), np.std(np.asarray(acc))))

    print('Finished program succesfully')


if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()