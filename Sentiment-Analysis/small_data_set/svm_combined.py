import sys
import os
import time
import nltk
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from sklearn.cross_validation import train_test_split
from nltk.corpus import nps_chat
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import nltk.corpus

from sklearn.svm import SVC,LinearSVC

stopwords = set(nltk.corpus.stopwords.words('english'))

short_pos = open("pos_tweet.txt","r").read()
short_neg = open("neg_tweet.txt","r").read()
short_neutral= open("neutral_tweet.txt","r").read()
test_f = open("test.txt","r").read()
#print (test_f)

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

for p in short_neutral.split('\n'):
    documents.append( ( "neutral", p) )
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

test_sen_all = ["Narendra modi is prime minister of India.",
"I am a boy.",
"Earth is full. Go home.",
"I dream that one day that I won't be as poor as a begger in uptown Manhattan."
"I work 40 hours a week to be this poor",
   " LOL as if its not good",
   "Most automated sentiment analysis tools are shit.",
   "VADER sentiment analysis is the shit.",
   "Sentiment analysis has never been good.",
   "Sentiment analysis with VADER has never been this good.",
   "Warren Beatty has never been so entertaining.",
   "I won't say that the movie is astounding and I wouldn't claim that \
   the movie is too banal either.",
   "I like to hate Michael Bay films, but I couldn't fault this one",
   "It's one thing to watch an Uwe Boll film, but another thing entirely \
   to pay for it",
   "The movie was too good",
   "This movie was actually neither that funny, nor super witty.",
   "This movie doesn't care about cleverness, wit or any other kind of \
   intelligent humor.",
   "Those who find ugly meanings in beautiful things are corrupt without \
   being charming.",
   "There are slow and repetitive parts, BUT it has just enough spice to \
   keep it interesting.",
   "The script is not fantastic, but the acting is decent and the cinematography \
   is EXCELLENT!",
   "Roger Dodger is one of the most compelling variations on this theme.",
   "Roger Dodger is one of the least compelling variations on this theme.",
   "Roger Dodger is at least compelling as a variation on the theme.",
   "they fall in love with the product",
   "but then it breaks",
   "usually around the time the 90 day warranty expires",
   "the twin towers collapsed today",
   "However, Mr. Carter solemnly argues, his client carried out the kidnapping \
   under orders and in the ''least offensive way possible.''"
 ]

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
    svm = SVC(C=1000000.0,  kernel='rbf')
    svm.fit(X, y)
    return svm

def train_svm_kernel_linear(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0,  kernel='linear')
    svm.fit(X, y)
    return svm

def train_linearSVM(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = LinearSVC()
    svm.fit(X, y)
    return svm
def train_svc_comb(x,y):
  svc=OneVsRestClassifier(LinearSVC(random_state=0)).fit(x, y)
  return svc
  
if __name__ == "__main__":
        
        X, y ,T = create_tfidf_training_data(documents,test_sen_all)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42 )

        
        
        svm_plane= train_svc_comb(X_train,y_train)
        #svm_linearkernel = train_svm_kernel_linear(X_train, y_train)
        #svm_linearSVM = train_linearSVM(X_train, y_train)
        #X_test = test_vectorizer(test_sen)
        #print (X_train , X_test)
        print(svm_plane.score(X_test, y_test))
        pred = svm_plane.predict(X_test)
        #print(confusion_matrix(pred, y_test))
        print(accuracy_score(y_test, pred ))
        print(classification_report(pred, y_test))
        pred = svm_plane.predict(T)
        print ((pred))
        



