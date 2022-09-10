#!/usr/bin/python

"""
Implement the Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.
Please check `viterbi.pl` for reference.

Usage:  python viterbi.py hmm-file < text > tags

special keywords:
 $init_state   (an HMM state) is the single, silent start state
 $final_state  (an HMM state) is the single, silent stop state
 $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word
"""

import sys, math
from collections import defaultdict

HMM_FILE = sys.argv[1]



init_state = "init"
final_state = "final"
OOV_symbol = "OOV"

A = defaultdict(lambda: defaultdict(lambda: 0))
B = defaultdict(lambda: defaultdict(lambda: 0))
States = defaultdict(lambda: 0) 
Voc = defaultdict(lambda: 0) 

#read in HMM file
with open(HMM_FILE) as HMMFile:
    for HMMString in HMMFile:
        HMMElements = HMMString.split()
        probType = HMMElements[0]  #first part of line, either trans or emit
        
        #for transmission probability:
        if probType == "trans":
            FROM = HMMElements[1]
            TO = HMMElements[2]
            P = float(HMMElements[3])
            
            #store log probability in dictionary A
            A[FROM][TO] = math.log(P) 
            
            #store states
            States[FROM] = 1
            States[TO] = 1
            
        #for emission probability:
        elif probType == "emit":
            STATE = HMMElements[1]
            SYMBOL = HMMElements[2]
            P = float(HMMElements[3])
            
            #store log probability in dictionary B
            B[STATE][SYMBOL] = math.log(P)
            
            #store state + symbol
            States[STATE] = 1
            Voc[SYMBOL] = 1
           

        

#read text file, line by line
for line in sys.stdin:
    w = line.split()
    n = len(w)
    V = defaultdict(lambda: defaultdict(lambda: 0))
    Backtrace = defaultdict(lambda: defaultdict(lambda: 0))
    
    V[0][init_state] = 0.0 #base case
    
    for i in range(1, n+1):
        word = w[i-1] #iterate over each word
        if word not in Voc: #if word not in voc, rename it with OOV symbol
            word = OOV_symbol
            
        for q in States: #each possible current state
            for qq in States: #each possible previous state
                if (q in A[qq]) and (word in B[q]) and (qq in V[i-1]): #only consider non-zeros
                    v = A[qq][q] + B[q][word] + V[i-1][qq]
                    if (i not in V) or (q not in V[i]) or (v > V[i][q]): #if found better previous state
                        V[i][q] = v #viturbi probability
                        Backtrace[i][q] = qq #best previous state
                        
    
    # this handles the last of the Viterbi equations, the one that brings
    # in the final state.
    
    foundgoal = 0
    goal = None
    q = None
    for qq in States: #for each possible state of the last word
        if (final_state in A[qq]) and (qq in V[n]):
            v = V[n][qq] + A[qq][final_state]
            if (foundgoal == 0) or (v > goal):
                goal = v
                foundgoal = 1
                q = qq
    
    
    #backtracking
    t = []
    for i in range(n, 0, -1):
        t.insert(0, q)
        q = Backtrace[i][q]
        
    
    if foundgoal:
        print(" ".join(t))
        
    else:
        print() #prints newline if HMM did not recognize 