# Methods read the XML files
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

import os
import json
import xml.etree.ElementTree as ET
from collections import Counter
import string
import en_core_web_sm
en_nlp = en_core_web_sm.load()
import nltk

from nltk.tokenize import RegexpTokenizer

import re
import numpy as np

def window(iterable, size):
    """
    Method obtained from Trusca et al. (2020). Sliding window.

    :param iterable:
    :param size:
    :return:
    """
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win

def _get_data_tuple(sptoks, asp_termIn, label, aspect_cat):
    """
    Method adapted from Trusca et al. (2020). Finds the ids of aspect terms, assigns a label to the senitment and to
    aspect categories.

    :param sptoks:
    :param asp_termIn:
    :param label:
    :param aspect_cat:
    :return:
    """
    aspect_is = []
    asp_term = ' '.join(sp for sp in asp_termIn).lower()
    for _i, group in enumerate(window(sptoks,len(asp_termIn))):
        if asp_term == ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i,_i+len(asp_termIn)))
            break
        elif asp_term in ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i,_i+len(asp_termIn)))
            break


    print(aspect_is)
    pos_info = []
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    lab = None
    if label == 'negative':
        lab = -1
    elif label == 'neutral':
        lab = 0
    elif label == "positive":
        lab = 1
    else:
        raise ValueError("Unknown label: %s" % lab)

    asp_cat = None
    if aspect_cat == 'AMBIENCE#GENERAL':
        asp_cat = 1
    if aspect_cat == 'DRINKS#PRICES':
        asp_cat = 2
    if aspect_cat == 'DRINKS#QUALITY':
        asp_cat = 3
    if aspect_cat == 'DRINKS#STYLE_OPTIONS':
        asp_cat = 4
    if aspect_cat == 'FOOD#PRICES':
        asp_cat = 5
    if aspect_cat == 'FOOD#QUALITY':
        asp_cat = 6
    if aspect_cat == 'FOOD#STYLE_OPTIONS':
        asp_cat = 7
    if aspect_cat == 'LOCATION#GENERAL':
        asp_cat = 8
    if aspect_cat == 'RESTAURANT#GENERAL':
        asp_cat = 9
    if aspect_cat == 'RESTAURANT#MISCELLANEOUS':
        asp_cat = 10
    if aspect_cat == 'RESTAURANT#PRICES':
        asp_cat = 11
    if aspect_cat == 'SERVICE#GENERAL':
        asp_cat = 12
    if aspect_cat == 'FOOD#GENERAL':
        asp_cat = 6

    return pos_info, lab, asp_cat


"""
Method obtained from Trusca et al. (2020). This function reads data from the xml file, we add punctuation removal.
NOTE: the punctuation removal can be easily avoided by commenting out some lines stressed in the code.
 
Iput arguments:
@fname: file location
@source_count: list that contains list [<pad>, 0] at the first position [empty input]
and all the unique words with number of occurences as tuples [empty input]
@source_word2idx: dictionary with unique words and unique index [empty input]
.. same for target
Return:
@source_data: list with lists which contain the sentences corresponding to the aspects saved by word indices 
@target_data: list which contains the indices of the target phrases: THIS DOES NOT CORRESPOND TO THE INDICES OF source_data 
@source_loc_data: list with lists which contains the distance from the aspect for every word in the sentence corresponding to the aspect
@target_label: contains the polarity of the aspect (0=negative, 1=neutral, 2=positive)
@max_sen_len: maximum sentence length
@max_target_len: maximum target length
"""
def read_data_2016(fname, source_count, source_word2idx, target_count, target_phrase2idx, file_name):
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)

    # parse xml file to tree
    tree = ET.parse(fname)
    root = tree.getroot()

    outF= open(file_name, "w")

    # save all words in source_words (includes duplicates)
    # save all aspects in target_words (includes duplicates)
    # finds max sentence length and max targets length
    source_words, target_words, max_sent_len, max_target_len = [], [], 0, 0
    target_phrases = []

    ## comment out to avoid removing punctuation
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    ##

    countConfl = 0
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)

        ## comment out to avoid removing punctuation
        sptoks = tokenizer.tokenize(sentenceNew)
        ##

        ## comment in to avoid removing punctuation
        # sptoks = nltk.word_tokenize(sentenceNew)
        ##

        for sp in sptoks:
            source_words.extend([''.join(sp).lower()])
        if len(sptoks) > max_sent_len:
            max_sent_len = len(sptoks)
        for opinions in sentence.iter('Opinions'):
            for opinion in opinions.findall('Opinion'):
                if opinion.get("polarity") == "conflict":
                    countConfl += 1
                    continue
                asp = opinion.get('target')

                if asp != 'NULL':
                    aspNew = re.sub(' +', ' ', asp)

                    ## comment in to avoid removing punctuation
                    # t_sptoks = nltk.word_tokenize(aspNew)
                    ##

                    ## comment out to avoid removing punctuation
                    t_sptoks = tokenizer.tokenize(aspNew)
                    ##

                    for sp in t_sptoks:
                        target_words.extend([''.join(sp).lower()])
                    target_phrases.append(' '.join(sp for sp in t_sptoks).lower())
                    if len(t_sptoks) > max_target_len:
                        max_target_len = len(t_sptoks)
    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())
    target_count.extend(Counter(target_phrases).most_common())

    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word] = len(source_word2idx)

    for phrase, _ in target_count:
        if phrase not in target_phrase2idx:
            target_phrase2idx[phrase] = len(target_phrase2idx)

    source_data, source_loc_data, target_data, target_label = list(), list(), list(), list()

    # collect output data (match with source_word2idx) and write to .txt file
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text

        ## comment in to avoid removing punctuation
        # sentenceNew = re.sub(' +', ' ', sent)
        # sptoks = nltk.word_tokenize(sentenceNew)
        ##

        ## comment out to avoid removing punctuation
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = tokenizer.tokenize(sentenceNew)
        ##

        if len(sptoks) != 0:
            idx = []
            for sptok in sptoks:
                idx.append(source_word2idx[''.join(sptok).lower()])
            for opinions in sentence.iter('Opinions'):
                for opinion in opinions.findall('Opinion'):
                    if opinion.get("polarity") == "conflict": continue
                    asp = opinion.get('target')
                    if asp != 'NULL': #removes implicit targets

                        ## comment in to avoid removing punctuation
                        # aspNew = re.sub(' +', ' ', asp)
                        # t_sptoks = nltk.word_tokenize(aspNew)
                        ##

                        ## comment out to avoid removing punctuation
                        aspNew = re.sub(' +', ' ', asp)
                        t_sptoks = tokenizer.tokenize(aspNew)
                        ##

                        source_data.append(idx)
                        outputtext = ' '.join(sp for sp in sptoks).lower()
                        outputtarget = ' '.join(sp for sp in t_sptoks).lower()
                        outputtext = outputtext.replace(outputtarget, '$T$')
                        outF.write(outputtext)
                        outF.write("\n")
                        outF.write(outputtarget)
                        outF.write("\n")
                        pos_info, lab, aspect_cat = _get_data_tuple(sptoks, t_sptoks, opinion.get('polarity'), opinion.get('category'))
                        pos_info = [(1-(i / len(idx))) for i in pos_info]
                        source_loc_data.append(pos_info)
                        targetdata = ' '.join(sp for sp in t_sptoks).lower()
                        target_data.append(target_phrase2idx[targetdata])
                        target_label.append(lab)
                        outF.write(str(lab))
                        outF.write("\n")
                        outF.write(str(aspect_cat))
                        outF.write("\n")

    outF.close()
    print("Read %s aspects from %s" % (len(source_data), fname))
    print(countConfl)
    return source_data, source_loc_data, target_data, target_label, max_sent_len, source_loc_data, max_target_len
