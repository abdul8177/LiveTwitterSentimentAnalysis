import nltk
import random  # shuffle up the dataset
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in words_features:
        features[w] = (w in words)  # if the top 3000 words is within the document it will return a boolean value
    return features


documents = []
all_words = []

pos = open("data/positive.txt", "r").read()
neg = open("data/negative.txt", "r").read()

allowed_word_types = ["J"]

for p in pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

words_features = list(all_words.keys())[:3000]

save_word_features = open("pickled_algos/word_features5k.pickle", "wb")
pickle.dump(words_features, save_word_features)
save_word_features.close()

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_featuresets = open("pickled_algos/featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)
# set that we'll train our classifier with
training_set = featuresets[:10000]

# set that we'll test against.
testing_set = featuresets[10000:]

# Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

# Naive Bayes Pickle
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# MNB
mnb_classifier = SklearnClassifier(MultinomialNB())
mnb_classifier.train(training_set)
print("MNB_Classifier accuracy percent:", (nltk.classify.accuracy(mnb_classifier, testing_set)) * 100)

# MNB Pickle
save_classifier = open("pickled_algos/MNB_classifier5k.pickle", "wb")
pickle.dump(mnb_classifier, save_classifier)
save_classifier.close()

# BernoulliNB
bnb_classifier = SklearnClassifier(BernoulliNB())
bnb_classifier.train(training_set)
print("BNB_Classifier accuracy percent:", (nltk.classify.accuracy(bnb_classifier, testing_set)) * 100)

# BernoulliNB pickle
save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle", "wb")
pickle.dump(bnb_classifier, save_classifier)
save_classifier.close()

# Logistic Regression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_Classifier accuracy percent:",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

# Logistic Regression PICKLE
save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

# SGD
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGD_Classifier accuracy percent:", (nltk.classify.accuracy(SGD_classifier, testing_set)) * 100)

# SGD PICKLE
save_classifier = open("pickled_algos/SGDC_classifier5k.pickle", "wb")
pickle.dump(SGD_classifier, save_classifier)
save_classifier.close()
#
# # SVC
# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)
#
# # SVC PICKLE
# save_classifier = open("pickled_algos/SVC_classifier5k.pickle", "wb")
# pickle.dump(SVC_classifier, save_classifier)
# save_classifier.close()

# LINEAR SVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

# LINEAR SVC PICKLE
save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

# NU SVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

# NU SVC PICKLE
save_classifier = open("pickled_algos/NuSVC_classifier5k.pickle", "wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGD_classifier,
                                  mnb_classifier,
                                  bnb_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

