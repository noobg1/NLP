from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import accuracy_score
import sys
import os
import time
import nltk
import random
import pickle
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
            
test_sen_all = []
for p in test_f.split('\n'):
    test_sen.append( p )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            test_sen_all.append(w[0].lower())

#print (test_sen[0], documents[0])

test_sen_all = ["I am in rvce where are you going",
"It is not good",
"It is not bad",
"Narendra modi is prime minister of India.",
"I am a boy.",
"Earth is full. Go home.",
"I dream that one day that I won't be as poor as a begger in uptown Manhattan.",
"I work 40 hours a week to be this poor",
   " LOL as if its not good",
   "Most automated sentiment analysis tools are shit.",
   "VADER sentiment analysis is the shit.",
   "Sentiment analysis has never been good.",
   "Sentiment analysis with VADER has never been this good.",
   "Warren Beatty has never been so entertaining.",
   "I won't say that the movie is astounding and I wouldn't claim that the movie is too banal either.",
   "I like to hate Michael Bay films, but I couldn't fault this one",
   "It's one thing to watch an Uwe Boll film, but another thing entirely to pay for it",
   "The movie was too good",
   "This movie was actually neither that funny, nor super witty.",
   "This movie doesn't care about cleverness, wit or any other kind of intelligent humor.",
   "Those who find ugly meanings in beautiful things are corrupt without being charming.",
   "There are slow and repetitive parts, BUT it has just enough spice to keep it interesting.",
   "The script is not fantastic, but the acting is decent and the cinematography is EXCELLENT!",
   "Roger Dodger is one of the most compelling variations on this theme.",
   "Roger Dodger is one of the least compelling variations on this theme.",
   "Roger Dodger is at least compelling as a variation on the theme.",
   "they fall in love with the product, but then it breaks usually around the time the 90 day warranty expires",
   "the twin towers collapsed today",
   "However, Mr. Carter solemnly argues, his client carried out the kidnapping under orders and in the ''least offensive way possible.''"
 ]
vectorizer = TfidfVectorizer(min_df=2,max_df=0.8,sublinear_tf=True,use_idf=True)
corpus_complete = [d[1] for d in documents]
temp=vectorizer.fit_transform(corpus_complete)
def create_tfidf_training_data(docs,test_docs):
    y = [d[0] for d in documents]

    # Create the document corpus list
    corpus = [d[1] for d in documents]
    #print (corpus[:3])
    # Create the TF-IDF vectoriser and transform the corpus
    # vectorizer = TfidfVectorizer(min_df=2,max_df=0.8,sublinear_tf=True,use_idf=True)
    X = vectorizer.fit_transform(corpus)

    test_corpus = test_docs
    #vectorizer = TfidfVectorizer()
    #print (test_corpus)
    T = vectorizer.transform(test_corpus)
    return X, y , T


def create_tfidf_training_data_pairwise(docs):
    y = [d[0] for d in docs]

    # Create the document corpus list
    corpus = [d[1] for d in documents]
    docs_corpus = [d[1] for d in docs]
    #print (corpus[:3])
    # Create the TF-IDF vectoriser and transform the corpus
    # vectorizer = TfidfVectorizer(min_df=2,max_df=0.8,sublinear_tf=True,use_idf=True)
    # temp = vectorizer.fit_transform(corpus)
    X = vectorizer.transform(docs_corpus)
    return X, y

def test_vectorizer(test_docs):

    test_corpus = [d for d in test_docs]

    # vectorizer = TfidfVectorizer()
    # print (test_corpus)
    X = vectorizer.transform(test_corpus)
    return X




def train_svm_plane(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

def train_svm_kernel_linear(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='linear')
    svm.fit(X, y)
    return svm

def train_linearSVM(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = LinearSVC()
    svm.fit(X, y)
    return svm


if __name__ == "__main__":
        
        # X, y = create_tfidf_training_data_pairwise(documents)

        # X_pos_neg, y_pos_neg  = create_tfidf_training_data_pairwise(documents_pos_neg)
        # X_pos_neut, y_pos_neut  = create_tfidf_training_data_pairwise(documents_pos_neut)
        # X_neg_neut, y_neg_neut   = create_tfidf_training_data_pairwise(documents_neg_neut)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42 )


        # X_neg_neut_train, X_neg_neut_test, y_neg_neut_train, y_neg_neut_test = train_test_split(X_neg_neut, y_neg_neut, test_size=0.1, random_state=42)
        # X_pos_neut_train, X_pos_neut_test, y_pos_neut_train, y_pos_neut_test = train_test_split(X_pos_neut, y_pos_neut, test_size=0.1, random_state=42 )
        # X_pos_neg_train, X_pos_neg_test, y_pos_neg_train, y_pos_neg_test = train_test_split(X_pos_neg, y_pos_neg, test_size=0.1, random_state=42 )
        
        # svm_plane_pos_neg= train_svm_plane(X_pos_neg_train, y_pos_neg_train)
        # svm_plane_pos_neut= train_svm_plane(X_pos_neut_train, y_pos_neut_train)
        # svm_plane_neg_neut= train_svm_plane(X_neg_neut_train, y_neg_neut_train)

        
        # svm_linearkernel_pos_neg = train_svm_kernel_linear(X_pos_neg_train, y_pos_neg_train)
        # svm_linearkernel_pos_neut = train_svm_kernel_linear(X_pos_neut_train, y_pos_neut_train)
        # svm_linearkernel_neg_neut = train_svm_kernel_linear(X_neg_neut_train, y_neg_neut_train)
        # # print ("type", type(svm_linearkernel_neg_neut))
        # svm_linearSVM_pos_neg = train_linearSVM(X_pos_neg_train, y_pos_neg_train)
        # svm_linearSVM_pos_neut = train_linearSVM(X_pos_neut_train, y_pos_neut_train)
        # svm_linearSVM_neg_neut = train_linearSVM(X_neg_neut_train, y_neg_neut_train)

        # with open('svm_linearSVM_pos_neg.pickle', 'wb') as fid:
        #     pickle.dump(svm_linearSVM_pos_neg, fid)
        # with open('svm_linearSVM_neg_neut.pickle', 'wb') as fid:
        #     pickle.dump(svm_linearSVM_neg_neut, fid)
        # with open('svm_linearSVM_pos_neut.pickle', 'wb') as fid:
        #     pickle.dump(svm_linearSVM_pos_neut, fid)
        
        open_file = open("svm_linearSVM_pos_neg.pickle", "rb")
        svm_lk_pos_neg = pickle.load(open_file)
        open_file.close()
        open_file = open("svm_linearSVM_pos_neut.pickle", "rb")
        svm_lk_pos_neut = pickle.load(open_file)
        open_file.close()
        open_file = open("svm_linearSVM_neg_neut.pickle", "rb")
        svm_lk_neg_neut = pickle.load(open_file)
        open_file.close()

        # pred1 = svm_linearSVM_pos_neg.predict(X_test)        

        # pred2 = svm_linearSVM_pos_neut.predict(X_test)  

        # pred3 = svm_linearSVM_neg_neut.predict(X_test)        

        # res=[]
        # for p1,p2,p3 in zip(pred1,pred2,pred3):
        #   if p1==p2:        
        #     res.append(p1)
        #   elif p1==p3:            
        #     res.append(p1)
        #   elif p2==p3:            
        #     res.append(p2)
        #   else:
        #     temp=[p1,p2,p3]
        #     res.append(random.choice(temp))

        # # print (len(res)," ",len(y_test))
        # print(accuracy_score(y_test, res))
        # print(classification_report(res, y_test))

        X_test = test_vectorizer(test_sen_all)
        

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
            res_predict.append("Couldnt predict")
        for t,r in zip(test_sen_all,res_predict):
          print (t," : ",r)