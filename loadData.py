# Methods to load the data
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

from dataReader2016 import read_data_2016
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random


def loadDataAndEmbeddings(config, loadData):
    """
    Method obtained from Trusca et al. (2020). Loads the data and the word emebeddings

    :param config:
    :param loadData:
    :return:
    """
    FLAGS = config

    if loadData == True:
        source_count, target_count = [], []
        source_word2idx, target_phrase2idx = {}, {}

        print('reading training data...')
        train_data = read_data_2016(FLAGS.train_data, source_count, source_word2idx, target_count, target_phrase2idx,
                                    FLAGS.train_path)
        print('reading test data...')
        test_data = read_data_2016(FLAGS.test_data, source_count, source_word2idx, target_count, target_phrase2idx,
                                   FLAGS.test_path)

        wt = np.random.normal(0, 0.05, [len(source_word2idx), 300])
        word_embed = {}
        count = 0.0
        with open(FLAGS.pretrain_file, 'r', encoding="utf8") as f:
            for line in f:
                content = line.strip().split()
                if content[0] in source_word2idx:
                    wt[source_word2idx[content[0]]] = np.array(list(map(float, content[1:])))
                    count += 1

        print('finished embedding context vectors...')

        # print data to txt file
        outF = open(FLAGS.embedding_path, "w")
        for i, word in enumerate(source_word2idx):
            outF.write(word)
            outF.write(" ")
            outF.write(' '.join(str(w) for w in wt[i]))
            outF.write("\n")
        outF.close()
        print((len(source_word2idx) - count) / len(source_word2idx) * 100)

        return len(train_data), len(test_data), train_data[4], test_data[4]

    else:
        # get statistic properties from txt file
        train_size, train_polarity_vector = getStatsFromFile(FLAGS.train_path)
        test_size, test_polarity_vector = getStatsFromFile(FLAGS.test_path)

        return train_size, test_size, train_polarity_vector, test_polarity_vector


def loadAverageSentence(config, sentences, pre_trained_context):
    """
    Method obtained from Trusca et al. (2020). Loads the average sentence.
    NOTE: not used in this project.

    :param config:
    :param sentences:
    :param pre_trained_context:
    :return:
    """
    FLAGS = config
    wt = np.zeros((len(sentences), FLAGS.edim))
    for id, s in enumerate(sentences):
        for i in range(len(s)):
            wt[id] = wt[id] + pre_trained_context[s[i]]
        wt[id] = [x / len(s) for x in wt[id]]

    return wt


def getStatsFromFile(path):
    """
    Method adapted from Trusca et al. (2020). Obtains the number of observations as well as the polarity vector

    :param path:
    :return:
    """
    polarity_vector = []
    with open(path, "r") as fd:
        lines = fd.read().splitlines()
        size = len(lines) / 4
        for i in range(0, len(lines), 4):
            # polarity
            polarity_vector.append(lines[i + 2].strip().split()[0])
    return size, polarity_vector


def loadHyperData(config, loadData, percentage=0.8):
    """
    Method adapted from Trusca et al. (2020). Loads data for hyperparameter tuning using validation set

    :param config:
    :param loadData:
    :param percentage:
    :return:
    """
    FLAGS = config

    if loadData:
        """Splits a file in 2 given the `percentage` to go in the large file."""
        random.seed(FLAGS.random_seed)
        with open(FLAGS.train_path, 'r') as fin, \
                open(FLAGS.hyper_train_path, 'w') as foutBig, \
                open(FLAGS.hyper_eval_path, 'w') as foutSmall:
            lines = fin.readlines()

            chunked = [lines[i:i + 4] for i in range(0, len(lines), 4)]
            random.shuffle(chunked)
            numlines = int(len(chunked) * percentage)
            for chunk in chunked[:numlines]:
                for line in chunk:
                    foutBig.write(line)
            for chunk in chunked[numlines:]:
                for line in chunk:
                    foutSmall.write(line)

    # get statistic properties from txt file
    train_size, train_polarity_vector = getStatsFromFile(FLAGS.hyper_train_path)
    test_size, test_polarity_vector = getStatsFromFile(FLAGS.hyper_eval_path)

    return train_size, test_size, train_polarity_vector, test_polarity_vector


def loadCrossValidation(config, split_size, load=True):
    """
    Method adapted from Trusca et al. (2020). Loads data for cross validation.
    NOTE: not used in this project and not tested.

    :param config:
    :param split_size:
    :param load:
    :return:
    """
    FLAGS = config
    if load:
        words, asp_cat, sent = [], [], []

        with open(FLAGS.train_path, encoding='cp1252') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                words.append([lines[i], lines[i + 1], lines[i + 2]], lines[i + 3])
                sent.append(lines[i + 2].strip().split()[0])
                asp_cat.append(lines[i + 3].strip().split()[0])
            words = np.asarray(words)
            asp_cat = np.asarray(asp_cat)
            sent = np.asarray(sent)

            i = 0
            kf = StratifiedKFold(n_splits=split_size, shuffle=True, random_state=FLAGS.random_seed)
            for train_idx, val_idx in kf.split(words, sent, asp_cat):
                words_1 = words[train_idx]
                words_2 = words[val_idx]
                with open("data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_train_' + str(
                        i) + '.txt', 'w') as train, \
                        open("data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_val_' + str(
                            i) + '.txt', 'w') as val:
                    for row in words_1:
                        train.write(row[0])
                        train.write(row[1])
                        train.write(row[2])
                        train.write(row[3])
                    for row in words_2:
                        val.write(row[0])
                        val.write(row[1])
                        val.write(row[2])
                        val.write(row[3])
                i += 1
        # get statistic properties from txt file
    train_size, train_polarity_vector = getStatsFromFile(
        "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_train_0.txt')
    test_size, test_polarity_vector = [], []
    for i in range(split_size):
        test_size_i, test_polarity_vector_i = getStatsFromFile(
            "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_val_' + str(i) + '.txt')
        test_size.append(test_size_i)
        test_polarity_vector.append(test_polarity_vector_i)

    return train_size, test_size, train_polarity_vector, test_polarity_vector