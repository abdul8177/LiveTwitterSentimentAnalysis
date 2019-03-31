from tweepy import  Stream, OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s
ckey = 'MF5d1dEgvieF4jtomRZNGVvbQ'
csecret = 'XGiPPMGN6TMWkoUeBLJV5zLuZ3cmpJ6zz7rYJv5UTA2yeAeLD2'
atoken = '389974581-oYfqlHBEWk0JANHP1hfv6np03fRSULcGyNcAYk8Z'
asecret = 'RAhy9eAzS24pWz1GCVe0myjmRs0ZESXrQJvlRyPI4JQrk'

# Listener class
class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data['text']
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet,  sentiment_value,  confidence)
        if confidence * 100 >= 80:
            output = open("output/twitter_out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return  True


    def on_error(self, status_code):
        print(status_code)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])