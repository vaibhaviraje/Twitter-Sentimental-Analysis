import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
from textblob.sentiments import NaiveBayesAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from cleantext import clean
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

from flask import Flask, render_template , redirect, url_for, request



def clean_tweet( tweet): 

        tweet=clean(tweet, no_emoji=True)
        tweet=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)|(#[A-Za-z0-9_]+)", " ", tweet).split())
        return tweet
         
def get_tweet_sentiment(tweet):
        

            MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL)
            
            encoded_text = tokenizer(tweet, return_tensors='pt')
            output = model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            scores_dict = {
            'Negative' : scores[0],
            'Neutral' : scores[1],
            'Positive' : scores[2]
            }
            keymax = max(zip( scores_dict.values(),scores_dict.keys()))
            confidenceval=str(round(keymax[0]*100,2))+'%'

            # sia = SentimentIntensityAnalyzer()
            # scores=sia.polarity_scores(tweet)
            # scores.pop("compound")
            # scores_dict={}
            # scores_dict['Negative']=scores['neg']
            # scores_dict['Neutral']=scores['neu']
            # scores_dict['Positive']=scores['pos']
            

            # keymax = max(zip( scores_dict.values(),scores_dict.keys()))
            # confidenceval=str(round(keymax[0]*100,2))+'%'
            return keymax[1]


        # analysis = TextBlob(clean_tweet(tweet)) 
        # if analysis.sentiment.polarity > 0:
        #     return "positive"
        # elif analysis.sentiment.polarity == 0:
        #     return "neutral"
        # else:
        #     return "negative"


def get_tweets(query, count=10): 
        
        count = int(count)
        tweets1 = [] 
        try: 
            
            client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAIsUjwEAAAAA0cdRmkkoHX6rHxB70tGbpPHSwJ4%3DK9fWAwGuG8NF3sBBcwqxolw3j1PTxSwr8BTmiLcwGPToMt3wRU')
            q = '#{} -is:retweet lang:en'.format(query)
            tweets = client.search_recent_tweets(query=q, tweet_fields=['context_annotations', 'created_at'], max_results=count)
            
           
            for tweet in tweets.data:
                t=tweet.text
                parsed_tweet = {} 
                parsed_tweet['text']=t
                parsed_tweet['sentiment']=get_tweet_sentiment(parsed_tweet['text']) 
            
                tweets1.append(parsed_tweet.copy())
            return tweets1 
        except tweepy.TweepyException as e: 
            print("Error : " + str(e)) 

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
  return render_template("index.html")

# ******Phrase level sentiment analysis
@app.route("/predict", methods=['POST','GET'])
def pred():
	if request.method=='POST':
            query=request.form['query']
            count=request.form['num']
            fetched_tweets = get_tweets(query, count) 
            return render_template('result.html', result=fetched_tweets)

# fetched_tweets
# [
#   {"text" : "tweet1", "sentiment" : "sentiment1"},
#   {"text" : "tweet2", "sentiment" : "sentiment2"},
#   {"text" : "tweet3", "sentiment" : "sentiment3"}
# ]

# *******Sentence level sentiment analysis
@app.route("/predict1", methods=['POST','GET'])
def pred1():
	if request.method=='POST':
            text = request.form['txt']
            MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL)
            
            encoded_text = tokenizer(text, return_tensors='pt')
            output = model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            scores_dict = {
            'Negative' : scores[0],
            'Neutral' : scores[1],
            'Positive' : scores[2]
            }
            keymax = max(zip( scores_dict.values(),scores_dict.keys()))
            confidenceval=str(round(keymax[0]*100,2))+'%'

            # sia = SentimentIntensityAnalyzer()
            # scores=sia.polarity_scores(text)
            # scores.pop("compound")
            # scores_dict={}
            # scores_dict['Negative']=scores['neg']
            # scores_dict['Neutral']=scores['neu']
            # scores_dict['Positive']=scores['pos']
            

            # keymax = max(zip( scores_dict.values(),scores_dict.keys()))
            # confidenceval=str(round(keymax[0]*100,2))+'%'
            
            return render_template('result1.html',msg=text, result=keymax[1],confidence=confidenceval)


if __name__ == '__main__':
    
    consumer_key = 'M85srcd9gPTola6ZUMAPAVdey' 
    consumer_secret = 'tGdDyyKJPnPm5Eih7s5oZsVidjLprHarDPhRlaR5JsT1KhXk8Y'
    access_token = '1596607128709439489-9QsX7R1Z5Q6Fna8iUAKQjyNTrZf4Pk'
    access_token_secret = '0XM8YnjHTghXhJ3xuIIC7ppSwga6V2A9ieAzzUHnYVwIB'

    try: 
        auth = OAuthHandler(consumer_key, consumer_secret)  
        auth.set_access_token(access_token, access_token_secret) 
        api = tweepy.API(auth)
    except: 
        print("Error: Authentication Failed") 

    app.debug=True
    app.run(host='localhost')

