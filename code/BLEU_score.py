import math



def BLEU_score(candidate, references, n):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on
    
    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.
    
    INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""

    candidate_list = candidate.split()
    reference_list = list(map(lambda x: x.split(), references))

    len_difference = float('inf')

    for reference in reference_list:
        if abs(len(reference) - len(candidate)) < abs(len_difference):
            len_difference = len(reference) - len(candidate)

    brevity = (len(candidate) + len_difference) / len(candidate)

    BP = 1 if brevity < 1 else math.exp(1 - brevity)

    p = 1.0

    for gram in range(n):
        not_appear = 0
        for i in range(len(candidate_list) - gram):
            s = ''
            for j in range(gram):
                s += (candidate_list[i + j] + ' ')

            for r in references:
                if r.find(s.strip()) >= 0:
                    continue
            not_appear += 1

        p *= ((len(candidate_list) - gram - not_appear) / (len(candidate_list) - gram))

    return BP * pow(p, 1 / n)
