from preprocess import *
import pickle
import os


def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """

    language_model = {'uni': {}, 'bi': {}}

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if file[-1] is language:
                fullFile = os.path.join(subdir, file)

                data = open(fullFile)

                for line in data:
                    line = line.rstrip('\n')
                    sentence = preprocess(line, language)

                    word_list = sentence.split(' ')

                    for i in range(len(word_list)):
                        word = word_list[i]

                        if word in language_model['uni']:
                            language_model['uni'][word] = language_model['uni'][word] + 1
                        else:
                            language_model['uni'][word] = 1

                        if i < len(word_list) - 1:
                            next_word = word_list[i + 1]

                            if word not in language_model['bi']:
                                language_model['bi'][word] = {}

                            if next_word in language_model['bi'][word]:
                                language_model['bi'][word][next_word] = language_model['bi'][word][next_word] + 1
                            else:
                                language_model['bi'][word][next_word] = 1
                data.close()

    # Save Model
    with open(fn_LM + '.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return language_model


# test
# lm_train('../data/Hansard/Trying/', 'e', 'english')
