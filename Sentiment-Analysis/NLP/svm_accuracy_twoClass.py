import sys
import os
import time
import nltk

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
documents = []
test_sen = []

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( ( "pos", p) )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( ( "neg", p) )
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



def create_tfidf_training_data(docs,test_docs):
    y = [d[0] for d in docs]

    # Create the document corpus list
    corpus = [d[1] for d in docs]
    #print (corpus[:3])
    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)

    test_corpus = test_docs

    #print (corpus[:5], test_corpus[:5])

    #vectorizer = TfidfVectorizer()
    #print (test_corpus)
    T = vectorizer.transform(test_corpus)
    return X, y , T


def test_vectorizer(test_docs):

    test_corpus = [d for d in test_docs]

    vectorizer = TfidfVectorizer()
    print (test_corpus)
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
        
        X, y ,T = create_tfidf_training_data(documents,test_sen_all)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42 )

        
        #svm = train_svm(X_train, y_train)
        svm_plane= train_svm_plane(X_train, y_train)
        svm_linearkernel = train_svm_kernel_linear(X_train, y_train)
        svm_linearSVM = train_linearSVM(X_train, y_train)

        #X_test = test_vectorizer(test_sen)
        #print (X_train , X_test)
        #pred = svm.predict(T)
        #print ((pred))
        print(svm_plane.score(X_test, y_test))
        pred = svm_plane.predict(X_test)
        print(confusion_matrix(pred, y_test))

        print(svm_linearkernel.score(X_test, y_test))
        pred = svm_linearkernel.predict(X_test)
        print(confusion_matrix(pred, y_test))

        print(svm_linearSVM.score(X_test, y_test))
        pred = svm_linearSVM.predict(X_test)
        print(confusion_matrix(pred, y_test))



