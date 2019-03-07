import re
import string


def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """

    modSent = in_sentence.lower().strip()

    modSent = re.sub(r"(!|\.|\?|,|:|;|\(|\)|-|\+|>|<|=|\")", r" \1 ", modSent)

    if language is 'f':
        modSent = re.sub(r"( l')(e|a)", r"\1 \2", modSent)
        modSent = re.sub(r" (b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|z)'(\w)", r" \1' \2", modSent)
        modSent = re.sub(r" (qu')(\w)", r" \1 \2", modSent)
        modSent = re.sub(r"(\w)'(on|il)", r"\1' \2", modSent)
        modSent = re.sub(r" d' (abord|accord|ailleurs|habitude) ", r" d'\1 ", modSent)

    modSent = 'SENTSTART ' + (re.sub(r" +", " ", modSent.strip())) + ' SENTEND'

    return modSent


# Testing
# print(preprocess("this is l'election and je t'aime and qu'on and puisqu'on and d'accord.", 'f'))
