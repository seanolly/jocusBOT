from twitterReader import Streamer
from authentication import consumer_key, consumer_secret, access_token, access_token_secret
##main

def run():
    streamer = Streamer(consumer_key, consumer_secret, access_token, access_token_secret)
    streamer.init(consumer_key, consumer_secret, access_token, access_token_secret)
    streamer.statuses.filter(track='#jocusBOT')
run()