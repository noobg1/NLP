import pandas as pd
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import classification_report,accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import random
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import time



df = pd.read_csv('training.csv')
sentiment = df['code']
content = df['tweet']
lmtzr = WordNetLemmatizer()

unique_sentiment = set(sentiment)
print (unique_sentiment)

data = []

stopwords = set(nltk.corpus.stopwords.words('english'))

def build_data(class_label, tweet):
	#  j is adject, r is adverb, and v is verb
	allowed_word_types = ["J","R","V"]
	
	words = word_tokenize(tweet)
	pos = nltk.pos_tag(words)
	chosen_word_list = ""
	for w in words:
		if w not in stopwords:
			chosen_word_list += (  lmtzr.lemmatize(w).lower())
			chosen_word_list += " "
	if chosen_word_list != " ":
   		data.append((class_label , chosen_word_list))
	return

def process_test_sentences(sentences):
	allowed_word_types = ["J","R","V"]
	test_sen = []
	for sen in sentences:
		words = word_tokenize(sen)
		pos = nltk.pos_tag(words)
		chosen_word_list = ""
		for w in words: 
			if  w not in stopwords:
				chosen_word_list += (  lmtzr.lemmatize(w).lower())
				chosen_word_list += " "
		if chosen_word_list != " ":
	   		test_sen.append( chosen_word_list)
	return test_sen


for i,j in zip(sentiment,content):
	if str(i) == '0':
		build_data("troubled", j)
	
	elif str(i) == '4':
		build_data("joyous", j)
	
	
	
	
	


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


vectorizer = TfidfVectorizer(min_df=2,max_df=0.9,sublinear_tf=True,use_idf=True)
corpus_complete = [d[1] for d in data]
temp=vectorizer.fit_transform(corpus_complete)
with open('pos_neg_large_dataset_linearSVM_corpus_transform_after_lemmatization.pickle', 'wb') as fi:
            pickle.dump(vectorizer, fi)

def create_tfidf_training_data_pairwise(docs):
    y = [d[0] for d in docs]

    # Create the document corpus list
    corpus = [d[1] for d in docs]
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

def train_linearSVM(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = LinearSVC()
    svm.fit(X, y)
    return svm

def train_svm_plane(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, kernel='rbf')
    svm.fit(X, y)
    return svm

if __name__ == "__main__":
        t1 = time.clock()
        random.shuffle(data)


        X, y = create_tfidf_training_data_pairwise(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42 )
        trained = train_linearSVM(X_train, y_train)
        with open('pos_neg_large_dataset_linearSVM_after_lemmatization.pickle', 'wb') as fid:
            pickle.dump(trained, fid)
        #X_test = test_vectorizer(test_sen_all)
        
        
        
        #trained = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)#svm_linearSVM.predict(X_test)
        predicted = trained.predict(X_test)
        # print(accuracy_score(y_test, predicted))
        # print(classification_report(y_test, predicted))

        # trained = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
        # predicted = trained.predict(X_test)
        #print (len(trained.decision_function(X_test)))
        print(accuracy_score(y_test, predicted))
        print(classification_report(y_test, predicted))
        # print(trained.predict_proba(X_test))
        # for _ in process_test_sentences(test_sen_all):
        #   print (_)
        X_test = test_vectorizer(process_test_sentences(test_sen_all))

        # print(accuracy_score(y_test, knnr))
        # print(classification_report(y_test, knnr))
        predicted = trained.predict(X_test)#trained.predict(X_test)

        for t,r in zip(test_sen_all,predicted):
          print (t," : ",r)

        print (time.clock() - t1)
        #print(neigh.predict(X_test))
       
