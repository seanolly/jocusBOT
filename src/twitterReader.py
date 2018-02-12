import numpy as np
from twython import TwythonStreamer
from twython import Twython
import itertools

from preprocessor import preprocessNormalTweets, preprocessJoke, jokeData, normalTweetData, labelData
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import GaussianNB
#initialize twitter account credentials
from sklearn.feature_selection import SelectPercentile

JOKE_CODE = 1
NOT_JOKE_CODE = 0

import stopwords as stopwords

import numpy as np
from twython import Twython
from sklearn.naive_bayes import MultinomialNB
from preprocessor import preprocessNormalTweets, preprocessJoke, jokeData, normalTweetData, labelData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import nltk.corpus
from nltk.stem import snowball
from sklearn.naive_bayes import GaussianNB


class Streamer(TwythonStreamer):

    DATA_LIMIT = 2000

    def init(self, consumer_key, consumer_secret, access_token, access_token_secret):
        classifier, features_train, labels_train, vectorizer = self.initClassifier()
        self.tweetBOT = Twython(consumer_key, consumer_secret, access_token, access_token_secret)
        self.classifier = classifier
        self.features_train = features_train
        self.labels_train = labels_train
        self.vectorizer = vectorizer

    ##initiate classifier, features, labels, and vectorizer.
    def initClassifier(self):
        features_train, labels_train, vectorizer = Streamer.initFeatures(self)
        classifier = GaussianNB()
        classifier.fit(features_train, labels_train)
        return classifier, features_train, labels_train, vectorizer

    ##preprocess training data
    def initFeatures(self):
        ### text vectorization--go from strings to lists of numbers
        # words to exclude
        exclusion = stopwords.get_stopwords('english')

        # vectorizer
        vectorizer = TfidfVectorizer(stop_words=exclusion)
        jokes = preprocessJoke(self.DATA_LIMIT, jokeData)
        jokeLabels = labelData(jokes, 1)  # 1 is joke
        print ("Joke Labels length:", len(jokeLabels))

        tweets = preprocessNormalTweets(self.DATA_LIMIT, normalTweetData)
        tweetLabels = labelData(tweets, 0)  # 0 is not joke
        print ("Tweet labels length:", len(tweetLabels))

        # concat joke_labels and tweet_labels
        training_labels = jokeLabels + tweetLabels
        print ("training label length:", len(training_labels))

        # concat features
        training_features = jokes + tweets
        print ("training feature length:", len(training_features))
        transformedFeatures = vectorizer.fit_transform(training_features).toarray()
        return transformedFeatures, training_labels, vectorizer

    ##transform tweets to vector, make prediction
    def makePrediction(self,classifier, vectorizer, tweets):
        transformedTweets = vectorizer.transform(tweets).toarray()
        prediction = classifier.predict((transformedTweets))
        return prediction

    def on_success(self, data):
        if 'text' in data and 'user' in data and 'screen_name' in data['user']:
            screen_name = data['user']['screen_name']
            content = data['text']
            prediction = self.makePrediction(self.classifier, self.vectorizer, [data['text']])
            self.predictionMSG(screen_name, prediction)
            print screen_name, content, prediction

    def on_error(self, status_code, data):
        print "ERROR!!!"
        print "STATUS: ", status_code, "DATA: ", data
        print(status_code)

    def predictionMSG(self, screen_name, prediction):
        message = "@" + screen_name + ": "
        if (prediction[0] == 0):
            message += " I don't think that was a joke. Or it wasn't funny at the very least."
            self.tweetBOT.update_status(status=message)
            print "NOT JOKE"
        else:
            message += " I think that was a joke. funny."
            self.tweetBOT.update_status(status=message)
            print "JOKE"