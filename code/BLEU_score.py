import math


def BLEU_score(candidate, references, n, brevity=False):
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

    appear = 0
    for i in range(len(candidate_list) - n + 1):
        s = ''
        for j in range(n):
            s += (candidate_list[i + j] + ' ')

        for r in references:
            if r.find(s.strip()) >= 0:
                appear += 1
                break

    p = (appear / (len(candidate_list) - n + 1))



    # calculate brevity
    if brevity:
        reference_list = list(map(lambda x: x.split(), references))
        len_difference = float('inf')

        for reference in reference_list:
            if abs(len(reference) - len(candidate_list)) < abs(len_difference):
                len_difference = len(reference) - len(candidate_list)

        brevity = (len(candidate_list) + len_difference) / len(candidate_list)

        BP = 1 if brevity < 1 else math.exp(1 - brevity)
        p = BP * pow(p, 1 / n)

    return p
