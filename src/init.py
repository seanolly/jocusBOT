from authentication import (consumer_key, consumer_secret, access_token, access_token_secret)
import numpy as np
from twython import Twython
import itertools

from preprocessor import preprocessNormalTweets, preprocessJoke, jokeData, normalTweetData, labelData
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import GaussianNB
#initialize twitter account credentials
twitter = Twython(consumer_key,
                  consumer_secret,
                  access_token,
                  access_token_secret)


import stopwords as stopwords

from authentication import (consumer_key, consumer_secret, access_token, access_token_secret)
import numpy as np
from twython import Twython
from sklearn.naive_bayes import MultinomialNB

from preprocessor import preprocessNormalTweets, preprocessJoke, jokeData, normalTweetData, labelData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import nltk.corpus
from nltk.stem import snowball
from sklearn.naive_bayes import GaussianNB
#initialize twitter account credentials
twitter = Twython(consumer_key,
                  consumer_secret,
                  access_token,
                  access_token_secret)



def initFeatures(limit):
    ### text vectorization--go from strings to lists of numbers
    # words to exclude
    exclusion = stopwords.get_stopwords('english')

    # vectorizer
    vectorizer = CountVectorizer(stop_words=exclusion)

    jokes = preprocessJoke(limit, jokeData)
    jokeLabels = labelData(jokes, 1) #1 is joke
    print ("Joke Labels length:", len(jokeLabels))

    tweets = preprocessNormalTweets(limit, normalTweetData)
    tweetLabels = labelData(tweets, 0) #0 is not joke
    print ("Tweet labels length:", len(tweetLabels))

    #concat joke_labels and tweet_labels
    training_labels = jokeLabels + tweetLabels
    print ("training label length:", len(training_labels))

    #concat features
    training_features = jokes + tweets
    print ("training feature length:", len(training_features))
    transformedFeatures = vectorizer.fit_transform(training_features).toarray()

    return transformedFeatures, training_labels, vectorizer

def makePrediction(vectorizer,features_train,labels_train, tweets):
    classifier = GaussianNB()
    classifier.fit(features_train, labels_train)
    transformedTweets = vectorizer.transform(tweets).toarray()
    print ("Transformed tweet: ", transformedTweets)
    prediction = classifier.predict((transformedTweets))
    return prediction



def run(data_limit):
    #get features, labels
    features_train, labels_train,vectorizer = initFeatures(2040)
    prediction = makePrediction(vectorizer, features_train, labels_train, ["The governor of England"])
    print prediction

run(1000)