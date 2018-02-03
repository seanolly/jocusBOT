import json
from sklearn.feature_extraction.text import CountVectorizer
data = json.load(open('reddit_jokes.json'))

vectorizer = CountVectorizer()

#preprocesses joke data
def preprocess(limit):
    result = []
    for i in range(0, limit):
        result.append(data[i]['title'] + " " + data[i]['body'])
    return result

result = preprocess(100)
vectorizer.fit(result)


