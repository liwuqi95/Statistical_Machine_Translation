import string
import os
from preprocess import *


# AM formatter


# A very useful tool to correct punctuation matching for the AM
def correct_punctuation(AM):
    # removing punctuation mapping

    for p in string.punctuation:
        if p in AM:
            del AM[p]

    for key in list(AM.keys()):
        for p in list(AM[key].keys()):
            if string.punctuation.find(p) >= 0:
                del AM[key][p]

    # rearrange the weight
    for key in AM:
        total = 0
        for k in AM[key]:
            total += AM[key][k]
        for k in AM[key]:
            AM[key][k] /= total

    for p in string.punctuation:
        AM[p] = {p: 1}

    return AM


# A tool to reduce the importance the unseen pair
def pay_appearance(train_dir, AM, num_sentences):
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
                english.append(processed_line.split())
            elif file[-1] == 'f':
                if len(french) >= num_sentences:
                    break
                processed_line = preprocess(line, 'f')
                french.append(processed_line.split())

    for (E, F) in zip(english, french):

        for e in E:

            for m in AM[e]:
                if m not in F:
                    AM[e][m] /= 1.5

    # rearrange the weight
    for key in AM:
        total = 0
        for k in AM[key]:
            total += AM[key][k]
        for k in AM[key]:
            AM[key][k] /= total

    return AM





















