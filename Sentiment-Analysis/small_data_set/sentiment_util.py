from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import accuracy_score
import sys
import os
import time
import nltk
import random
import pickle
import itertools
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from nltk.corpus import nps_chat
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC,LinearSVC



short_pos = open("pos_tweet.txt","r").read()
short_neg = open("neg_tweet.txt","r").read()
short_neutral= open("neutral_tweet.txt","r").read()
test_f = open("test.txt","r").read()

# move this up here
all_words = []
documents=[]
documents_pos_neg = []
documents_pos_neut = []
documents_neg_neut = []
test_sen = []

#  j is adject, r is adverb, and v is verb
allowed_word_types = ["J","R","V"]
# allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( ( "pos", p) )
    documents_pos_neut.append( ( "pos", p) )
    documents_pos_neg.append( ( "pos", p) )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( ( "neg", p) )
    documents_pos_neg.append( ( "neg", p) )
    documents_neg_neut.append( ( "neg", p) )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neutral.split('\n'):
    documents.append( ( "neutral", p) )
    documents_pos_neut.append( ( "neutral", p) )
    documents_neg_neut.append( ( "neutral", p) )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            


test_sen_all = ["I am in rvce where are you going"]
vectorizer = TfidfVectorizer(min_df=2,max_df=0.8,sublinear_tf=True,use_idf=True)
corpus_complete = [d[1] for d in documents]
temp=vectorizer.fit_transform(corpus_complete)


def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

def test_vectorizer(test_docs):

    test_corpus = [d for d in test_docs]    
    X = vectorizer.transform(test_corpus)
    return X




def sentiment(message):
        
        
        open_file = open("svm_linearSVM_pos_neg.pickle", "rb")
        svm_lk_pos_neg = pickle.load(open_file)
        open_file.close()
        open_file = open("svm_linearSVM_pos_neut.pickle", "rb")
        svm_lk_pos_neut = pickle.load(open_file)
        open_file.close()
        open_file = open("svm_linearSVM_neg_neut.pickle", "rb")
        svm_lk_neg_neut = pickle.load(open_file)
        open_file.close()

        

        X_test = test_vectorizer(message)
        

        pred1 = svm_lk_pos_neg.predict(X_test)        

        pred2 = svm_lk_pos_neut.predict(X_test)  

        pred3 = svm_lk_neg_neut.predict(X_test)

        res=[]
        res_predict=[]
        for p1,p2,p3 in zip(pred1,pred2,pred3):
          if p1==p2:        
            res.append(p1)
            res_predict.append(p1)
          elif p1==p3:            
            res.append(p1)
            res_predict.append(p1)
          elif p2==p3:            
            res.append(p2)
            res_predict.append(p2)
          else:
            temp=[p1,p2,p3]
            res.append(random.choice(temp))
            res_predict.append("meh")

        
        return (most_common(res_predict))

