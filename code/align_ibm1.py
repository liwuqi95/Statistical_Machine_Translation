from lm_train import *
from log_prob import *
from preprocess import *
import pickle
import os


def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    eng, fre = read_hansard(train_dir, num_sentences)

    AM = initialize(eng, fre)

    for iter in range(max_iter):
        AM = em_step(AM, eng, fre)

    AM['SENTSTART'] = {'SENTSTART': 1}
    AM['SENTEND'] = {'SENTEND': 1}

    with open(fn_AM + '.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM


# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
    files = os.listdir(train_dir)
    files.sort()

    english = []
    french = []

    for file in files:
        if files.count(file[0:-1] + 'e') + files.count(file[0:-1] + 'f') is not 2:
            continue
        opened_file = open(train_dir + file, "r")
        for line in opened_file:
            if file[-1] == 'e':
                if len(english) >= num_sentences:
                    break
                processed_line = preprocess(line, 'e')
                english.append(processed_line.split(' '))
            elif file[-1] == 'f':
                if len(french) >= num_sentences:
                    break
                processed_line = preprocess(line, 'f')
                french.append(processed_line.split(' '))

        opened_file.close()
    return english, french


def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    AM = {}

    for i in range(len(eng)):
        eng_sentence = eng[i]
        fre_sentence = fre[i]

        for j in range(1, len(eng_sentence) - 1):
            if eng_sentence[j] not in AM:
                AM[eng_sentence[j]] = {}

            for k in range(1, len(fre_sentence) - 1):
                if fre_sentence[k] not in AM[eng_sentence[j]]:
                    AM[eng_sentence[j]][fre_sentence[k]] = 1.0

    for i in AM:
        v = len(AM[i])

        for j in AM[i]:
            AM[i][j] = 1 / v

    return AM


def em_step(t, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""

    tcount = {}
    total = {}
    for i in t:
        tcount[i] = {}
        total[i] = 0.0
        for j in t[i]:
            tcount[i][j] = 0.0

    for E, F in zip(eng, fre):
        for f in list(set(F[1:-1])):
            denom_c = 0
            eset = list(set(E[1:-1]))
            for e in eset:
                denom_c += t[e][f] * F.count(f)
            for e in eset:
                sub = t[e][f] * F.count(f) * E.count(e) / denom_c
                tcount[e][f] += sub
                total[e] += sub

    for e in t:
        for f in t[e]:
            t[e][f] = tcount[e][f] / total[e]

    return t

