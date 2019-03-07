from preprocess import *
from lm_train import *
from math import log2


def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""

    word_list = sentence.split(' ')

    prob = 1.0

    for i in range(len(word_list) - 1):
        word = word_list[i]
        next = word_list[i + 1]

        word_count = LM['uni'][word] if (word in LM['uni']) else 0
        next_count = LM['bi'][word][next] if (word in LM['bi']) and (next in LM['bi'][word]) else 0

        if smoothing:
            next_count += delta
            word_count += delta * vocabSize

        prob_i = (next_count / word_count) if word_count > 0 else 0.0

        if prob_i == 0.0:
            return float('-inf')
        else:
            prob *= prob_i

    return log2(prob)
