#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle
import os
import decode
from align_ibm1 import *
from BLEU_score import *
from lm_train import *
from preprocess import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'

discussion = """
Discussion :

{As n increases, the BLEU scores decrease since it adding more requirement for the translation.
 As we add more sentences, most of the scores increase due to increasing amount of data. 
 However, some of the scores decrease with more sentences, it may due to over fitting. }
"""


##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """

    if use_cached and os.path.isfile(fn_LM + '.pickle'):
        return pickle.load(open(fn_LM + '.pickle', "rb"))
    else:
        return lm_train(data_dir, language, fn_LM)


def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    if use_cached and os.path.isfile(fn_AM + '.pickle'):
        return pickle.load(open(fn_AM + '.pickle', "rb"))
    else:
        return align_ibm1(data_dir, num_sent, max_iter, fn_AM)


def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """

    references = eng + google_refs
    score = []

    for eng in eng_decoded:
        p = 1
        for i in range(1, n + 1):
            p *= BLEU_score(eng, references, i, i == n)
        score.append(p)

    return score


def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """

    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    f = open("Task5.txt", 'w+')
    f.write(discussion)
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    train_dir = '../data/Hansard/Training/'

    LM = _getLM(train_dir, 'e', '../cache/LM')

    # AM_names = [1000, 10000, 15000, 30000]
    AM_names = [1000, 10000, 15000, 30000]
    AMs = []
    for num_sent in AM_names:
        AMs.append(_getAM(train_dir, num_sent, 10, f"../cache/{num_sent}_AM"))

    for i, AM in enumerate(AMs):

        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")

        eng_decoded = []
        eng = []
        google_refs = []

        data = open("../data/Hansard/Testing/Task5.f", "r")
        for line in data:
            eng_decoded.append(decode.decode(preprocess(line, 'f'), LM, AM))

        data = open("../data/Hansard/Testing/Task5.e", "r")
        for line in data:
            eng.append(preprocess(line, 'e'))

        data = open("../data/Hansard/Testing/Task5.google.e", "r")
        for line in data:
            google_refs.append(preprocess(line, 'e'))

        all_evals = []
        for n in range(1, 4):
            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            evals = _get_BLEU_scores(eng_decoded, eng, google_refs, n)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()

    main(args)
