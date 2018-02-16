from twitterReader import Streamer
from authentication import consumer_key, consumer_secret, access_token, access_token_secret
##main

def run():
    print "setting up streamer"
    streamer = Streamer(consumer_key, consumer_secret, access_token, access_token_secret)
    print "initiating classifier, training..."
    streamer.init(consumer_key, consumer_secret, access_token, access_token_secret)
    print "filtering tweets..."
    streamer.statuses.filter(track='#jocusBOT')

run()