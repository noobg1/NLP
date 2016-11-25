import nltk
import random
from stemming.porter2 import stem
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

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

# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]

# print(documents[0][0])
documents=[]
temp=""
temp_list=[]
word_features=[]
all_words=[]
with open("pos_tweet") as f:
  content = f.readlines()
#print (content)
for c in content:
  temp=c.strip()
  temp=temp.split(" ")
  temp1=[]
  for w in temp :
    if w in stop:
      continue
    w=stem(w)

    w=w.lower()
    if w[0].isdigit():
      w=""
    elif w[0]=='@':
      w=""
    elif "http" in w:
      w=""
    all_words.append(w)
    word_features.append(w)    
    temp1=temp1+list(w)
    
  # print (temp)  
  documents.append((temp1,'pos'))

with open("neg_tweet") as f:
  content = f.readlines()
# print (content)
for c in content:
  temp=c.strip()
  temp=temp.split(" ")
  temp1=[]
  for w in temp :
    if w in stop:
      continue
    w=stem(w)
    w=w.lower()
    if w[0].isdigit():
      w=""
    elif w[0]=='@':
      w=""
    elif "http" in w:
      w=""
    word_features.append(w)
    temp1=temp1+list(w)
  # print (temp)  
  documents.append((temp1,'neg'))

# print (word_features)
#print (documents)
random.shuffle(documents)



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


if __name__ == "__main__":
        
        X, y ,T = create_tfidf_training_data(documents,test_sen_all)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42 )

        
        
        svm_plane= train_svm_plane(X,y)
        #svm_linearkernel = train_svm_kernel_linear(X_train, y_train)
        #svm_linearSVM = train_linearSVM(X_train, y_train)

        #X_test = test_vectorizer(test_sen)
        #print (X_train , X_test)
        pred = svm_plane.predict(T)
        print ((pred))
        
