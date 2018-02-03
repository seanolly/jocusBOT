import json
from sklearn.feature_extraction.text import TfidfVectorizer
jokeData = json.load(open('reddit_jokes.json'))
normalTweetData = json.load(open('short.json'))

#returns array of jokes
def preprocessJoke(limit, jokeData):
    result = []
    for i in range(0, limit):
        result.append(jokeData[i]['title'] + " " + jokeData[i]['body'])
    return result

#returns array of tweets
def preprocessNormalTweets(limit , tweetData):
    result = []
    for i in range(0, limit):
        result.append(tweetData[i]['content'])
    return result

def labelData(data, label):
    result = []
    for i in range(0, len(data)):
        result.append(label)
    return result



