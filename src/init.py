from authentication import (consumer_key, consumer_secret, access_token, access_token_secret)
from twython import Twython

#initialize twitter account credentials
twitter = Twython(consumer_key,
                  consumer_secret,
                  access_token,
                  access_token_secret)

twitter.update_status()