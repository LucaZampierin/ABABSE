# Hyperparameter tuning using Tree-structure Parzen Estimator (TPE)
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

#import parameter configuration and data paths
from config import *

#import modules
import random
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, atpe
import numpy as np
import sys
import pickle
import os
import traceback
from bson import json_util
import json
import matplotlib.pyplot as plt

train_size, test_size, train_polarity_vector, test_polarity_vector = loadHyperData(FLAGS, False)

# Define variabel spaces for hyperopt to run over
eval_num = 0
best_loss = None
best_hyperparams = None

ababse1space = [
                hp.choice('learning_rate',[0.001, 0.005, 0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
                hp.quniform('drop_rate', 0, 0.6, 0.05),
                hp.choice('beta1',    [0.9, 0.92, 0.95, 0.97, 0.99 ]),
                hp.choice('beta2',    [ 0.9, 0.92, 0.95, 0.97, 0.99 ]),
                hp.choice('l2',    [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.04, 0.07, 0.1 ]),
                hp.choice('seed_reg',    [ 0.001, 0.01, 0.1, 1, 5, 10 ]),
                hp.choice('ortho_reg',    [ 0.001, 0.01, 0.1, 1, 5, 10 ]),
                hp.choice('batchsize',    [ 10, 20, 30, 40, 60 ]),
                hp.choice('neg_samples',    [ 5, 10, 15, 20 ])
            ]

ababse2space = [
                hp.choice('learning_rate',[0.001, 0.005, 0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
                hp.quniform('drop_rate', 0, 0.6, 0.05),
                hp.choice('beta1',    [0.9, 0.92, 0.95, 0.97, 0.99 ]),
                hp.choice('beta2',    [ 0.9, 0.92, 0.95, 0.97, 0.99 ]),
                hp.choice('l2',    [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.04, 0.07, 0.1 ]),
                hp.choice('seed_reg',    [ 0.001, 0.01, 0.1, 1, 5, 10 ]),
                hp.choice('ortho_reg',    [ 0.001, 0.01, 0.1, 1, 5, 10 ]),
                hp.choice('batchsize',    [ 10, 20, 30, 40, 60 ]),
                hp.choice('neg_samples',    [ 5, 10, 15, 20 ])
            ]

lcrunsspace = [
                hp.choice('learning_rate',[0.001, 0.005, 0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
                hp.quniform('drop_rate', 0, 0.6, 0.05),
                hp.choice('beta1',    [0.9, 0.92, 0.95, 0.97, 0.99 ]),
                hp.choice('beta2',    [ 0.9, 0.92, 0.95, 0.97, 0.99 ]),
                hp.choice('l2',    [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.04, 0.07, 0.1 ]),
                hp.choice('seed_reg',    [ 0.001, 0.01, 0.1, 1, 5, 10 ]),
                hp.choice('ortho_reg',    [ 0.001, 0.01, 0.1, 1, 5, 10 ]),
                hp.choice('batchsize',    [ 10, 20, 30, 40, 60 ]),
                hp.choice('neg_samples',    [ 5, 10, 15, 20 ])
            ]

# Define objectives for hyperopt
def ababse1_objective(hyperparams):
    """
    Method adapted from Trusca et al. (2020). Runs ABABSE1 using different hyperparameters.

    :param hyperparams: (learning rate, dropout rate, beta1, beta2, L2 regularization, seed regularization,
    orthogonal regularization, batchsize, number of negative samples).
    :return:
    """
    global eval_num
    global best_loss
    global best_hyperparams

    eval_num += 1
    (learning_rate, drop_rate, beta1, beta2, l2, seed_reg, ortho_reg, batchsize, neg_samples) = hyperparams
    print(hyperparams)

    vocab_pos = {'good': 0, 'great': 1, 'nice': 2, 'impecable': 3, 'excellent': 4}
    vocab_neg = {'gross': 0, 'bad': 1, 'terrible': 2, 'awful': 3, 'horrible': 4}
    vocab_neu = {'mediocre': 0, 'reasonable': 1, 'decent': 2, 'average': 3, 'ok': 4}
    sub_vocab = [ vocab_pos, vocab_neu, vocab_neg]
    tf.reset_default_graph()
    min_loss, best_epoch, acc, pred1, att, true, pr, rec, f1, tr_acc, sent_emb, per_class_scores = \
        ABABSE1.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, test_size, sub_vocab, learning_rate, drop_rate, beta1,
                   beta2, l2, seed_reg, ortho_reg, batchsize, neg_samples)
    tf.reset_default_graph()

    # Save training results to disks with unique filenames
    print(eval_num, min_loss, hyperparams)

    if best_loss is None or min_loss < best_loss:
        best_loss = min_loss
        best_hyperparams = hyperparams

    result = {
            'loss':   min_loss,
            'status': STATUS_OK,
            'space': hyperparams,
            'epoch': best_epoch,
            'accuracy': acc
        }
    save_json_result('ABABSE1' +str(FLAGS.year) +'acc' +str(acc) + 'loss'+ str(min_loss), result)
    return result


def ababse2_objective(hyperparams):
    """
    Method adapted from Trusca et al. (2020). Runs ABABSE2 using different hyperparameters.

    :param hyperparams: (learning rate, dropout rate, beta1, beta2, L2 regularization, seed regularization,
    orthogonal regularization, batchsize, number of negative samples).
    :return:
    """
    global eval_num
    global best_loss
    global best_hyperparams

    eval_num += 1
    (learning_rate, drop_rate, beta1, beta2, l2, seed_reg, ortho_reg, batchsize, neg_samples) = hyperparams
    print(hyperparams)

    vocab_pos = {'good': 0, 'great': 1, 'nice': 2, 'impecable': 3, 'excellent': 4}
    vocab_neg = {'gross': 0, 'bad': 1, 'terrible': 2, 'awful': 3, 'horrible': 4}
    vocab_neu = {'mediocre': 0, 'reasonable': 1, 'decent': 2, 'average': 3, 'ok': 4}
    sub_vocab = [ vocab_pos, vocab_neu, vocab_neg]
    tf.reset_default_graph()
    min_loss, best_epoch, acc, pred1, att, true, pr, rec, f1, tr_acc, sent_emb, per_class_scores = \
        ABABSE2.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, test_size, sub_vocab, learning_rate, drop_rate, beta1,
                   beta2, l2, seed_reg, ortho_reg, batchsize, neg_samples)
    tf.reset_default_graph()

    # Save training results to disks with unique filenames
    print(eval_num, min_loss, hyperparams)

    if best_loss is None or min_loss < best_loss:
        best_loss = min_loss
        best_hyperparams = hyperparams

    result = {
            'loss':   min_loss,
            'status': STATUS_OK,
            'space': hyperparams,
            'epoch': best_epoch,
            'accuracy': acc
        }
    save_json_result('ABABSE2' +str(FLAGS.year) +'acc' +str(acc) + 'loss'+ str(min_loss), result)
    return result


# Define objectives for hyperopt
def lcruns_objective(hyperparams):
    """
    Method adapted from Trusca et al. (2020). Runs Uns-LCR-Rot using different hyperparameters.

    :param hyperparams: (learning rate, dropout rate, beta1, beta2, L2 regularization, seed regularization,
    orthogonal regularization, batchsize, number of negative samples).
    :return:
    """
    global eval_num
    global best_loss
    global best_hyperparams

    eval_num += 1
    (learning_rate, drop_rate, beta1, beta2, l2, seed_reg, ortho_reg, batchsize, neg_samples) = hyperparams
    print(hyperparams)

    vocab_pos = {'good': 0, 'great': 1, 'nice': 2, 'impecable': 3, 'excellent': 4}
    vocab_neg = {'gross': 0, 'bad': 1, 'terrible': 2, 'awful': 3, 'horrible': 4}
    vocab_neu = {'mediocre': 0, 'reasonable': 1, 'decent': 2, 'average': 3, 'ok': 4}
    sub_vocab = [ vocab_pos, vocab_neu, vocab_neg]
    tf.reset_default_graph()
    min_loss, best_epoch, acc, pred1, fw1, bw1, tl1, tr1, true, pr, rec, f1, tr_acc, sent_emb, per_class_scores = \
        lcrModelU.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, test_size, sub_vocab, learning_rate, drop_rate,
                       beta1, beta2, l2, seed_reg, ortho_reg, batchsize, neg_samples)
    tf.reset_default_graph()

    # Save training results to disks with unique filenames
    print(eval_num, min_loss, hyperparams)

    if best_loss is None or l < best_loss:
        best_loss = l
        best_hyperparams = hyperparams

    result = {
            'loss':   min_loss,
            'status': STATUS_OK,
            'space': hyperparams,
            'epoch': best_epoch,
            'accuracy': acc
        }
    save_json_result('lcruns' +str(FLAGS.year) +'acc' +str(acc) + 'loss'+ str(min_loss), result)
    return result


# Run a hyperopt trial
def run_a_trial():
    """
    Method obtained from Trusca et al. (2020). Runs one iteration of the TPE algorithm

    :return:
    """
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        # Insert the method opbjective funtion
        ababse1_objective,
        # Define the methods hyperparameter space
        space     = ababse1space,
        algo      = tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("results.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")
    print(best_hyperparams)

def print_json(result):
    """
    Method obtained from Trusca et al. (2020). Prints a .json file

    :param result:
    :return:
    """
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))

def save_json_result(model_name, result):
    """
    Method obtained from Trusca et al. (2020). Save json to a directory and a filename.

    :param model_name:
    :param result:
    :return:
    """
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists("results/"):
        os.makedirs("results/")
    with open(os.path.join("results/", result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name):
    """
    Method obtained from Trusca et al. (2020). Load json from a path.
    :param best_result_name:
    :return:
    """
    result_path = os.path.join("results/", best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
        )

def load_best_hyperspace():
    """
    Method obtained from Trusca et al. (2020). Loads the current best hyperparameters.
    NOTE: not used in this project.

    :return:
    """
    results = [
        f for f in list(sorted(os.listdir("results/"))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)["space"]

def plot_best_model():
    """
    Method obtained from Trusca et al. (2020). Plots the best model found yet.
    NOTE: not used in this project.

    :return:
    """
    space_best_model = load_best_hyperspace()
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    print("Best hyperspace yet:")
    print_json(space_best_model)

while True:
    print("Optimizing New Model")
    try:
        run_a_trial()
    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
    plot_best_model()
