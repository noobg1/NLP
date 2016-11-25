import sys

import numpy
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance
import nltk.corpus
from nltk import decorators
import nltk.stem
from nltk.corpus import nps_chat
posts = nps_chat.xml_posts()

stemmer_func = nltk.stem.snowball.EnglishStemmer().stem
stopwords = set(nltk.corpus.stopwords.words('english'))

@decorators.memoize
def normalize_word(word):
    return stemmer_func(word.lower())

def get_words(titles):
    words = set()
    for title in job_titles:
        for word in title.split():
            words.add(normalize_word(word))
    return list(words)

@decorators.memoize
def vectorspaced(title):
    title_components = [normalize_word(word) for word in title.split()]
    return numpy.array([
        word in title_components and not word in stopwords
        for word in words], numpy.short)

if __name__ == '__main__':

    filename = 'example.txt'
    if len(sys.argv) == 2:
        filename = sys.argv[1]

    with open(filename) as title_file:

        job_titles = []
        for sen in posts[:300]:
            job_titles.append(str(sen.text))

        #print type(posts)
        #job_titles = [line.strip() for line in title_file.readlines()]

        words = get_words(job_titles)

        # cluster = KMeansClusterer(5, euclidean_distance)
        cluster = GAAClusterer(5)
        cluster.cluster([vectorspaced(title) for title in job_titles if title])

        # NOTE: This is inefficient, cluster.classify should really just be
        # called when you are classifying previously unseen examples!
        classified_examples = [
                cluster.classify(vectorspaced(title)) for title in job_titles
            ]

        for cluster_id, title in sorted(zip(classified_examples, job_titles)):
            print (cluster_id, title)