#!/usr/bin/python

"""
Implement a trigram HMM and viterbi here.
You model should output the final tags similar to `viterbi.pl`.

Usage:  python trigram_hmm.py train_tags train_text inference_text > tags

"""

import sys, math, re
import numpy as np
from collections import defaultdict, deque

OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

HMM_FILE = "trigram_hmm.out"

def deleted_interpolation(uni_c, bi_c, tri_c):

    lambda1 = 0
    lambda2 = 0
    lambda3 = 0

    #iterate over all trigrams
    for a in tri_c:
        for b in tri_c[a]:
            for c in tri_c[a][b]:
                #making sure it's a valid trigram
                if (tri_c[a][b][c]) > 0:

                    #three possibilities
                    c1 = 0
                    if (bi_c[a][b] - 1 > 0):
                        c1 = float(tri_c[a][b][c] - 1) / (bi_c[a][b] - 1)
                    c2 = 0
                    if (uni_c[c] - 1) > 0:
                        c2 = float(bi_c[b][c] - 1) / (uni_c[c] - 1)
                    c3 = 0
                    if (sum(uni_c.values()) - 1) > 0 :
                        c3 = float(uni_c[c] - 1) / (sum(uni_c.values()) - 1)

                    #take maximum out of the three
                    k = np.argmax([c1, c2, c3])

                    #add to corresponding lambda
                    if k == 0:
                        lambda3 += tri_c[a][b][c]
                    elif k == 1:
                        lambda2 += tri_c[a][b][c]
                    elif k == 2:
                        lambda1 += tri_c[a][b][c]


    #normalize weights
    weights = [lambda1, lambda2, lambda3]
    normalized = [float(Lambda) / sum(weights) for Lambda in weights]
    return normalized


def train_trigram_hmm(tag_file, text_file):

    """
    Inputs:
        tag_file: training file containing tags
        text_file: training file containing text

    Outputs:
        A: transmission probability, dictionary (3-dim)
            A[prevprevTag][prevTag][tag]
        B: emission probability, dictionary (2-dim)
            B[tag][word]
    """

    #n-gram counts (of tags)
    tri_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    bi_counts = defaultdict(lambda: defaultdict(lambda: 0))
    uni_counts = defaultdict(lambda: 0)

    #emission counts
    e_counts = defaultdict(lambda: defaultdict(lambda: 0))
    e_total = defaultdict(lambda: 0)

    #total number of tokens
    N = 0

    #transmission/emission probabilities
    #log probabilities
    A = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    B = defaultdict(lambda: defaultdict(lambda: float('-Inf')))
    #non-log probabilities
    A_nl = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    B_nl = defaultdict(lambda: defaultdict(lambda: 0))

    #vocab and possible tags
    vocab = defaultdict(lambda: 0)
    allTags = defaultdict(lambda: 0)

    #add first and last tags to taglist
    allTags[INIT_STATE] = 1
    allTags[FINAL_STATE] = 1

    #get counts
    with open(tag_file) as TagFile, open(text_file) as TextFile:
        for tagString, tokenString in zip(TagFile, TextFile):   #for each line
            tags = re.split("\s+", tagString.rstrip())
            tokens = re.split("\s+", tokenString.rstrip())
            pairs = list(zip(tags, tokens))

            #starting two states are start states
            prevprevTag = INIT_STATE
            prevTag = INIT_STATE

            for (tag, word) in pairs:

                if word not in vocab: #gets emission counts for OOV words
                    vocab[word] = 1
                    word = OOV_WORD

                #update n-gram counts of tags
                tri_counts[prevprevTag][prevTag][tag] += 1
                bi_counts[prevTag][tag] += 1
                uni_counts[tag] += 1

                #update emission counts
                e_counts[tag][word] += 1
                e_total[tag] += 1

                #update total number of tokens
                N += 1

                #increment tag pointers
                prevprevTag = prevTag
                prevTag = tag

                #update taglist
                allTags[tag] = 1

            #counts for end of sentence
            tri_counts[prevprevTag][prevTag][FINAL_STATE] += 1
            bi_counts[prevTag][FINAL_STATE] += 1
            uni_counts[FINAL_STATE] += 1
            N += 1


    #get deleted interp weights
    intWeights = deleted_interpolation(uni_counts, bi_counts, tri_counts)

    #do interp for transition probabilities
    for prevprevTag in allTags:
        for prevTag in allTags:
            for tag in allTags:

                if (tag != INIT_STATE): #won't ever have case where the third word is INIT_STATE

                    #calculate trigram prob
                    triP = 0
                    if bi_counts[prevprevTag][prevTag] != 0:
                        triP = float(tri_counts[prevprevTag][prevTag][tag]) / bi_counts[prevprevTag][prevTag]

                    #bigram prob
                    biP = 0
                    if uni_counts[prevTag] != 0:
                        biP = float(bi_counts[prevTag][tag]) / uni_counts[prevTag]

                    #unigram prob
                    uniP = float(uni_counts[tag]) / N

                    probs = [triP, biP, uniP]

                    #dot prod with interpolation weights
                    A[prevprevTag][prevTag][tag] = math.log(np.dot(intWeights, probs))
                    A_nl[prevprevTag][prevTag][tag] = np.dot(intWeights, probs)
    """
    #naive trigram
    for prevprevTag in allTags:
        for prevTag in allTags:
            for tag in allTags:
                biC = bi_counts[prevprevTag][prevTag]
                triC = tri_counts[prevprevTag][prevTag][tag]
                if (biC != 0) and (triC != 0):
                    A[prevprevTag][prevTag][tag] = math.log(float(triC)/biC)
                else:
                    A[prevprevTag][prevTag][tag] = 0

    """


    #vocab present in emission matrix (will be diff. from vocab logged during counts due to handling of OOV)
    actual_vocab = defaultdict(lambda: str)

    #get emission probabilities
    for tag in e_counts:
        for word in e_counts[tag]:
            B[tag][word] = math.log(float(e_counts[tag][word]) / e_total[tag]) #log prob
            B_nl[tag][word] = float(e_counts[tag][word]) / e_total[tag]        #non-log prob (for output)
            actual_vocab[word] = 1




    return A, A_nl, B, B_nl, list(set(actual_vocab)), list(set(allTags))






def viterbi(textfile, vocab, alltags, t, e):

    #read in text file
    with open(textfile) as textFile:

        #one line at a time...
        linenum = 1
        for line in textFile:

            #split line into words
            w = line.split()
            n = len(w)       #num words
            V = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float("-Inf"))))  #Viterbi probability
            Backtrace = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))      #backtracing

            V[0][INIT_STATE][INIT_STATE] = 0.0 #base case

            for i in range(1, n+1): #from 1 to n
                word = w[i-1]

                if word not in vocab:
                    #print(word, "not in vocab")
                    word = OOV_WORD

                for tag in alltags: #over all possible states
                    for prevTag in alltags: #over all possible previous states
                        for prevprevTag in alltags: #over all possible prev prev states
                            if (prevTag in V[i-1][prevprevTag]) \
                                and (tag in t[prevprevTag][prevTag]) and (word in e[tag]):
                                v = t[prevprevTag][prevTag][tag] + e[tag][word] + V[i-1][prevprevTag][prevTag]
                                #if this prob is better than previous (or first one)
                                if (i not in V) or (prevTag not in V[i]) or \
                                (tag not in V[i][prevTag]) or (v > V[i][prevTag][tag]):
                                    V[i][prevTag][tag] = v
                                    Backtrace[i][prevTag][tag] = (prevprevTag, prevTag)




            #to find best path to FINAL_STATE
            foundgoal = 0
            goal = None
            pp = None
            p = None

            #over all possible prevprev and prev tags
            for prevprevTag in alltags:
                for prevTag in alltags:
                    if (V[n][prevprevTag][prevTag] != 0): #if viterbi prob. of last column has prevprev and prev
                        v = V[n][prevprevTag][prevTag] + t[prevprevTag][prevTag][FINAL_STATE] #no emission prob for final
                        if (foundgoal == 0) or (v > goal):
                            foundgoal = 1
                            goal = v
                            pp = prevprevTag
                            p = prevTag



            if foundgoal:
                #backtrace
                tagged = []
                for i in range(n, 0, -1):
                    tagged.insert(0, p)
                    pair = Backtrace[i][pp][p]
                    #increment prevprev and prev
                    pp = pair[0]
                    p = pair[1]
                print(" ".join(tagged))

            else:
                print() #empty line if final state transition not found

            #print("line", linenum, "done")

            linenum += 1


    return





if __name__ == "__main__":

    #filenames from command line input
    TAG_FILE = sys.argv[1]
    TEXT_FILE = sys.argv[2]
    INF_FILE = sys.argv[3]

    #train hmm
    trans, t_nl, emit, e_nl, vocab, alltags = train_trigram_hmm(TAG_FILE, TEXT_FILE)

    #write hmm probabilities to file (non-log probabilities)
    with open(HMM_FILE, 'w') as HMMOut:
        for prevprevTag in trans:
                for prevTag in trans[prevprevTag]:
                    for tag in trans[prevprevTag][prevTag]:
                        HMMOut.write(("trans %s %s %s %s\n" % (prevprevTag, prevTag, tag, float(t_nl[prevprevTag][prevTag][tag]))))
        for tag in emit:
            for word in emit[tag]:
                HMMOut.write(("emit %s %s %s\n" % (tag, word, float(e_nl[tag][word]))))



    viterbi(INF_FILE,vocab, alltags, trans, emit)
