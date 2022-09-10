import sys
import getopt
import os
import math
import operator
import numpy as np

from collections import defaultdict, Counter

"""
FUNCTIONS I WROTE:
- class NaiveBayes: init(), classify(), addExample()
- analyze_model

Everything else was code given to us in the hmwk brief! 
"""


"""
Improvement experimentations:

1. Stop-word filtering + binary NB
    I first tried implementing stop-word filtering with my binary NB model, and got an accuracy of 0.827000, which was slightly
    worse than binary NB without stop-word filtering.

2. Punctuation filtering + binary NB
    Then, since I noticed there was a lot of punctuation cropping up in the pos/neg class words lists, I tried filtering out
    punctuation + binary NB, and got an accuracy of 0.830000, so slightly better!

3. Stop-word and punctuation filtering + binary NB
    I also tried using both stop-word filtering and punctuation filtering with binary NB, and got an accuracy of 0.828000 - so,
    stop-word filtering wasn't helping much.

4. Punctuation filtering + binary NB + laplace alpha tuning
    Finally, I tried a few different values of alpha in the laplace smoothing for binary NB:
    alpha = 2, accuracy = 0.834000
    alpha = 3, accuracy = 0.836000
    alpha = 4, accuracy = 0.839000
    alpha = 5, accuracy = 0.840500
    alpha = 6, accuracy = 0.839500
    alpha = 5.5, accuracy = 0.840500
    alpha = 5.75, accuracy = 0.841000



FINAL BEST MODEL:
- punctuation filtering
- binary NB
- laplace smoothing: alpha = 5.75

accuracy = 0.841

"""

class NaiveBayes:
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
        """
        def __init__(self):
          self.train = []
          self.test = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
           words is a list of strings.
        """
        def __init__(self):
            self.klass = ''
            self.words = []


    def __init__(self):
        """NaiveBayes initialization"""
        self.FILTER_STOP_WORDS = False
        self.BOOLEAN_NB = False
        self.BEST_MODEL = False
        self.stopList = set(self.readFile('data/english.stop'))
        self.numFolds = 10

        self.vocab = defaultdict(lambda: 0)


        # frequency of the class
        self.freq_class = defaultdict(lambda: 0)

        # frequency of words
        self.freq_words = defaultdict(lambda: 0)

        # frequency of words per class
        self.freq_pos_words = defaultdict(lambda: 0)
        self.freq_neg_words = defaultdict(lambda: 0)

        #words in each document
        self.document_words = []

        #labels for each document
        self.document_labels = []

        self.doc_count_words_pos = defaultdict(lambda: 0)
        self.doc_count_words_neg = defaultdict(lambda: 0)

        # number of words per class (|V| for C)

    #############################################################################
    # TODO TODO TODO TODO TODO
    # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
    # Boolean (Binarized) features.
    # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
    # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
    # that relies on feature counts.
    #
    # If the BEST_MODEL flag is true, include your new features and/or heuristics that
    # you believe would be best performing on train and test sets.
    #
    # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the
    # other two are meant to be off. That said, if you want to include stopword removal
    # or binarization in your best model, write the code accordingl

    def classify(self, words):
        """ TODO
            'words' is a list of words to classify. Return 'pos' or 'neg' classification.
        """
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)

        if self.BOOLEAN_NB:
            #implement boolean nb
            total_docs = self.freq_class['pos'] + self.freq_class['neg']
            pos_label_prob = self.freq_class['pos'] / total_docs
            neg_label_prob = self.freq_class['neg'] / total_docs



            sum_dc_all_words_pos = sum(self.doc_count_words_pos.values())
            sum_dc_all_words_neg = sum(self.doc_count_words_neg.values())


            #get pos prob
            pos_prob = 0
            for word in set(words):
                #if word in self.doc_count_words_pos:
                pos_prob += math.log((self.doc_count_words_pos[word] + 1) / (sum_dc_all_words_pos + len(self.vocab)*1))
               # else:
                    #pos_prob += 1
            pos_prob += math.log(pos_label_prob)

            #get neg prob
            neg_prob = 0
            for word in set(words):
                #if word in self.doc_count_words_neg:
                neg_prob += math.log((self.doc_count_words_neg[word] + 1) / (sum_dc_all_words_neg + len(self.vocab)*1))
                #else:
                    #neg_prob += 1
            neg_prob += math.log(neg_label_prob)

            if pos_prob > neg_prob:
                return 'pos'
            else:
                return 'neg'

        elif self.BEST_MODEL:
            #implement filtering punctuation
            #and then use laplace smoothing with alpha = 5.75


            punctuationList = ["'", ".", "!", ",", "/", "?", "(", ")", "*", '"']
            filtered = []
            for word in words:
                if not word in punctuationList and word.strip() != '':
                    filtered.append(word)

            words = filtered


            total_docs = self.freq_class['pos'] + self.freq_class['neg']
            pos_label_prob = self.freq_class['pos'] / total_docs
            neg_label_prob = self.freq_class['neg'] / total_docs



            sum_dc_all_words_pos = sum(self.doc_count_words_pos.values())
            sum_dc_all_words_neg = sum(self.doc_count_words_neg.values())


            #get pos prob
            pos_prob = 0
            for word in set(words):
                #if word in self.doc_count_words_pos:
                pos_prob += math.log((self.doc_count_words_pos[word] + 5.75) / (sum_dc_all_words_pos + len(self.vocab)*5.75))
               # else:
                    #pos_prob += 1
            pos_prob += math.log(pos_label_prob)

            #get neg prob
            neg_prob = 0
            for word in set(words):
                #if word in self.doc_count_words_neg:
                neg_prob += math.log((self.doc_count_words_neg[word] + 5.75) / (sum_dc_all_words_neg + len(self.vocab)*5.75))
                #else:
                    #neg_prob += 1
            neg_prob += math.log(neg_label_prob)

            if pos_prob > neg_prob:
                return 'pos'
            else:
                return 'neg'

        else:
            #implement naive
            total_docs = self.freq_class['pos'] + self.freq_class['neg']
            pos_label_prob = self.freq_class['pos'] / total_docs
            neg_label_prob = self.freq_class['neg'] / total_docs

            all_pos_words = sum(self.freq_pos_words.values())
            all_neg_words = sum(self.freq_neg_words.values())

            total_vocab = len(self.freq_pos_words) + len(self.freq_neg_words)


            # positive likelihood

            pos_prob = 0
            for word in words:
                #pos_prob += math.log(self.freq_pos_words[word] + 1 / (all_pos_words + total_vocab)) # laplace smoothing
                pos_prob += math.log((self.freq_pos_words[word] + 1) / (all_pos_words + len(self.vocab)*1))
            pos_prob += math.log(pos_label_prob)


            # negative likelihood

            neg_prob = 0
            for word in words:
                #neg_prob += math.log(self.freq_neg_words[word] + 1 / (all_neg_words + total_vocab)) # laplace smoothing
                neg_prob += math.log((self.freq_neg_words[word] + 1) / (all_neg_words + len(self.vocab)*1))
            neg_prob += math.log(neg_label_prob)


            if pos_prob > neg_prob:
                return 'pos'
            else:
                return 'neg'


    def addExample(self, klass, words):
        """
         * TODO
         * Train your model on an example document with label klass ('pos' or 'neg') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier
         * in the NaiveBayes class.
         * Returns nothing
        """
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)

        # frequency of the class
        self.freq_class[klass] += 1


        if self.BOOLEAN_NB:
            for word in set(words):
                self.freq_words[word] += 1
                self.vocab[word] = 1
                if klass == 'pos':
                    self.doc_count_words_pos[word] += 1
                else:
                    self.doc_count_words_neg[word] += 1




        elif self.BEST_MODEL:


            punctuationList = ["'", ".", "!", ",", "/", "?", "(", ")", "*", '"']
            filtered = []
            for word in words:
                if not word in punctuationList and word.strip() != '':
                    filtered.append(word)

            words = filtered


            for word in set(words):
                self.freq_words[word] += 1
                self.vocab[word] = 1
                if klass == 'pos':
                    self.doc_count_words_pos[word] += 1
                else:
                    self.doc_count_words_neg[word] += 1

        else:
            for word in words:
                self.vocab[word] = 1
                self.freq_words[word] += 1
                if klass == 'pos':
                    self.freq_pos_words[word] += 1
                else:
                    self.freq_neg_words[word] += 1







    # END TODO (Modify code beyond here with caution)
    #############################################################################


    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here,
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))
        return result


    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()


    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        for fileName in posTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            example.klass = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            example.klass = 'neg'
            split.train.append(example)
        return split

    def train(self, split):
        for example in split.train:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words =  self.filterStopWords(words)
            self.addExample(example.klass, words)


    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        #for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posTrainFileNames:
              example = self.Example()
              example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
              example.klass = 'pos'
              if fileName[2] == str(fold):
                  split.test.append(example)
              else:
                  split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            yield split


    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for example in split.test:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words =  self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels

    def buildSplits(self, args):
        """Builds the splits for training/testing"""
        trainData = []
        testData = []
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print('[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir))

            posTrainFileNames = os.listdir('%s/pos/' % trainDir)
            negTrainFileNames = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posTrainFileNames:
                    example = self.Example()
                    example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    example.klass = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(example)
                    else:
                        split.train.append(example)
                for fileName in negTrainFileNames:
                    example = self.Example()
                    example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    example.klass = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(example)
                    else:
                        split.train.append(example)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print('[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir))
            posTrainFileNames = os.listdir('%s/pos/' % trainDir)
            negTrainFileNames = os.listdir('%s/neg/' % trainDir)
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                example.klass = 'pos'
                split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                split.train.append(example)

            posTestFileNames = os.listdir('%s/pos/' % testDir)
            negTestFileNames = os.listdir('%s/neg/' % testDir)
            for fileName in posTestFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                example.klass = 'pos'
                split.test.append(example)
            for fileName in negTestFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                example.klass = 'neg'
                split.test.append(example)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    classifier = None
    for split in splits:
        classifier = NaiveBayes()
        classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
        classifier.BOOLEAN_NB = BOOLEAN_NB
        classifier.BEST_MODEL = BEST_MODEL
        accuracy = 0.0
        for example in split.train:
            words = example.words
            classifier.addExample(example.klass, words)

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0
        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy))
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print('[INFO]\tAccuracy: %f' % avgAccuracy)

    # interpret the decision rule of the model of the last fold
    pos_signal_words, neg_signal_words = analyze_model(classifier)
    print('[INFO]\tWords for pos class: %s' % ','.join(pos_signal_words))
    print('[INFO]\tWords for neg class: %s' % ','.join(neg_signal_words))


def classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    classifier.BEST_MODEL = BEST_MODEL
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print(classifier.classify(testFile))

def main():
    FILTER_STOP_WORDS = False
    BOOLEAN_NB = False
    BEST_MODEL = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f','') in options:
        FILTER_STOP_WORDS = True
    elif ('-b','') in options:
        BOOLEAN_NB = True
    elif ('-m','') in options:
        BEST_MODEL = True

    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, args[0], args[1])
    else:
        test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL)


def analyze_model(nb_classifier):
    # TODO: This function takes a <nb_classifier> as input, and outputs two word list <pos_signal_words> and
    #  <neg_signal_words>. <pos_signal_words> is a list of 10 words signaling the positive klass, and <neg_signal_words>
    #  is a list of 10 words signaling the negative klass.



    if nb_classifier.BOOLEAN_NB or nb_classifier.BEST_MODEL:

        sum_dc_all_words_pos = sum(nb_classifier.doc_count_words_pos.values())
        sum_dc_all_words_neg = sum(nb_classifier.doc_count_words_neg.values())

        max_pos_words = defaultdict(lambda: 0)
        max_neg_words = defaultdict(lambda: 0)
        for word in nb_classifier.vocab:
            pos_prob = (nb_classifier.doc_count_words_pos[word] / sum_dc_all_words_pos)
            neg_prob = (nb_classifier.doc_count_words_neg[word] / sum_dc_all_words_neg)
            max_pos_words[word] = pos_prob - neg_prob
            max_neg_words[word] = neg_prob - pos_prob

        sorted_max_pos_words = sorted(max_pos_words.items(), key=lambda x:x[1], reverse=True)
        sorted_max_neg_words = sorted(max_neg_words.items(), key=lambda x:x[1], reverse=True)

        neg_signal_words = []
        pos_signal_words = []

        for i in range(10):
            neg_signal_words.append(sorted_max_neg_words[i][0])
            pos_signal_words.append(sorted_max_pos_words[i][0])

        return pos_signal_words, neg_signal_words

    else:
        all_pos_words = sum(nb_classifier.freq_pos_words.values())
        all_neg_words = sum(nb_classifier.freq_neg_words.values())

        max_pos_words = defaultdict(lambda: 0)
        max_neg_words = defaultdict(lambda: 0)
        for word in nb_classifier.vocab:
            pos_prob = (nb_classifier.freq_pos_words[word]) / (all_pos_words)
            neg_prob = (nb_classifier.freq_neg_words[word]) / (all_neg_words)
            max_pos_words[word] = pos_prob - neg_prob
            max_neg_words[word] = neg_prob - pos_prob

        sorted_max_pos_words = sorted(max_pos_words.items(), key=lambda x:x[1], reverse=True)
        sorted_max_neg_words = sorted(max_neg_words.items(), key=lambda x:x[1], reverse=True)

        neg_signal_words = []
        pos_signal_words = []


        for i in range(10):
            neg_signal_words.append(sorted_max_neg_words[i][0])
            pos_signal_words.append(sorted_max_pos_words[i][0])

        return pos_signal_words, neg_signal_words


if __name__ == "__main__":
    main()
