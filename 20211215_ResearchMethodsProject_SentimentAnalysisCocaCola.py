import re
import tweepy
from tweepy.errors import TweepyException
from tweepy import OAuthHandler 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from wordcloud import WordCloud,STOPWORDS
from textblob import TextBlob 
import datetime
import numpy as np
import statistics
from scipy import stats as st

def connect():
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_token_secret = ''
    
    try:
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        return api
    except:
        print("Error Connection to Twitter API")
    
def cleanText(text):
    text = text.lower()
    # Removes all mentions (@username) from the tweet since it is of no use to us
    text = re.sub(r'(@[A-Za-z0-9_]+)', '', text)
      
    # Removes any link in the text
    text = re.sub('http://\S+|https://\S+', '', text)
    
    # Only considers the part of the string with char between a to z or digits and whitespace characters
    # Basically removes punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Removes stop words that have no use in sentiment analysis 
    text_tokens = word_tokenize(text)
    text = [word for word in text_tokens if not word in stopwords.words()]
    
    text = ' '.join(text)
    return text

def stem(text):
    # This function is used to stem the given sentence
    porter = PorterStemmer()
    token_words = word_tokenize(text)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    return " ".join(stem_sentence)

def sentiment(cleaned_text):
    # Returns the sentiment based on the polarity of the input TextBlob object
    if cleaned_text.sentiment.polarity > 0:
        return 'positive'
    elif cleaned_text.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def fetch_tweets(api, query, startDate, count = 100):    
    tweets = [] # Empty list that stores all the tweets
    
    fetched_data = []
    
    # Fetches the tweets using the api
    try:   
        toDate = startDate + datetime.timedelta(hours = 24)
        
        for i in range(9):
            fetched_data.extend(api.search_full_archive(
									label = 'research', 
									query = query, 
									fromDate = startDate.strftime("%Y%m%d%H%M"), 
									toDate = toDate.strftime("%Y%m%d%H%M"), 
									maxResults = count)
									)
			
            startDate = startDate + datetime.timedelta(hours = 24)
            toDate = toDate + datetime.timedelta(hours = 24)   
                  
        for tweet in fetched_data:
            txt = tweet.text
            clean_txt = cleanText(txt) # Cleans the tweet
            stem_txt = TextBlob(stem(clean_txt)) # Stems the tweet
            sent = sentiment(stem_txt) # Gets the sentiment from the tweet 
            tweets.append((txt, clean_txt, sent, stem_txt.polarity))
        
        return tweets
    
    except TweepyException as e:
        print("Error" + str(e))
        pass

def CreateChartMostUsedWords(tweets, numberOfWords, plotTitle):      
    # We use a dictionary to store all the words found in tweets.
    # As value we store the number of appearances of each word.
    # Dictionary is a data-structure that doesn't allow duplicates as keys.    
    wordDictionary = {}
    
    # We start to iterate in data and take every row (tweet)
    for row in tweets['clean_tweets']:
        
		# Finding a float or int value on a row would return an error
		# For this case we check every row and if it is not a string we make it string
        if (isinstance(row, str)):
            row = row
        else:            
            row = str(row)
        
        # We split the row in a list of words.
        words = row.split(' ')
        
        # We take word by word
        for word in words:
            
            # We check if the word is present in the dictionary
            if word not in wordDictionary:
                # Word is not present so we add it with initial number of appearances equal to 1.
                wordDictionary[word] = 1
            else:
                # Word is presesnt in the dictionary so we increment the word count.
                wordDictionary[word] = wordDictionary[word] + 1
                
    # Remove RT word
    wordDictionary.pop("rt")
    
    # We need to sort the dictionary by the number of word appearances.
    sorted_dict = dict(sorted(wordDictionary.items(),
                           key=lambda item: item[1],
                           reverse=True))
    
    # We take the 35 most used words from dictionary and add them to a list
    firstItems = list(sorted_dict.items())[:numberOfWords]
    
    # We create a new DataFrame from the resulted list
    wordsDataFrame = pd.DataFrame(firstItems)
    
    # We add the columns' names
    wordsDataFrame.columns = ["Word", "Number of appearances"]
    
    # We set the index to be the Word column
    wordsDataFrame.set_index('Word',inplace=True)
    
    # We make the final plot
    wordsDataFrame.plot(kind = "barh"
           ,title = f"Most used {numberOfWords} words {plotTitle}"
           ,figsize=(5,10)
           ,color = 'red'
           )
    
print ("Starting...")
# Authentication on Twitter
api = connect() # Gets the tweepy API object

tenDaysBeforeConferenceDateTime = datetime.datetime(2021, 6, 4, 15, 0)
pressConferenceDateTime = datetime.datetime(2021, 6, 14, 15, 0)

print ("Getting before event tweets...")
# tweetsBefore = fetch_tweets(api,'#cocacola lang:en', tenDaysBeforeConferenceDateTime, tenDaysBeforeConferenceDateTime, 100)
tweetsBefore = pd.read_csv("./New_BeforeWithPolarity.csv")

print ("Getting after event tweets...")
# tweetsAfter = fetch_tweets(api,'#cocacola lang:en', pressConferenceDateTime, pressConferenceDateTime, 100)
tweetsAfter = pd.read_csv("./New_AfterWithPolarity.csv")

print ("Creating the DataFrames for both tweets...")
dfBefore = pd.DataFrame(tweetsBefore, columns= ['tweets', 'clean_tweets','sentiment', 'score'])
dfAfter = pd.DataFrame(tweetsAfter, columns= ['tweets', 'clean_tweets','sentiment', 'score'])

# print ("Eliminating duplicates from cleaned tweets...")
# dfBefore = dfBefore.drop_duplicates(subset='clean_tweets')
# dfAfter = dfAfter.drop_duplicates(subset='clean_tweets')

# print ("Saving in CSV files...")
# dfBefore.to_csv('New_BeforeWithPolarity.csv', index= False)
# dfAfter.to_csv('New_AfterWithPolarity.csv', index= False)

print ("Ploting Most used 35 words before event")
CreateChartMostUsedWords(dfBefore, 35, 'before')

print ("Ploting Most used 35 words after event")
CreateChartMostUsedWords(dfAfter, 35, 'after')


dfBefore['clean_tweets'] = dfBefore['clean_tweets'].str.replace('rt', '')
dfAfter['clean_tweets'] = dfAfter['clean_tweets'].str.replace('rt', '')

print ("Creating WordCloud for Before words")
twt = " ".join([str(elem) for elem in dfBefore['clean_tweets']])
wordcloud1 = WordCloud(stopwords=STOPWORDS, background_color='white', width=2500, height=2000).generate(twt)

plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()


print ("Creating WordCloud for After words")
twt2 = " ".join([str(elem) for elem in dfAfter['clean_tweets']])
wordcloud2 = WordCloud(stopwords=STOPWORDS, background_color='white', width=2500, height=2000).generate(twt2)

plt.figure(2,figsize=(13, 13))
plt.imshow(wordcloud2)
plt.axis('off')
plt.show()


print ("Creating plot with stock fluctuation")
# Define the ticker symbol
tickerSymbol = 'KO'

# Get data on this ticker
cokeStocksBeforeAndAfterDataFrame = yf.download(tickerSymbol, start='2021-6-1', end='2021-7-14')
cokeStocksBeforeAndAfterDataFrame['Adj Close'].plot()

print ("Calculating sentiment analysis...")
#Before
ptweets_before = dfBefore[dfBefore['sentiment'] == 'positive']
p_perc_before = 100 * len(ptweets_before)/len(tweetsBefore)

ntweets_before = dfBefore[dfBefore['sentiment'] == 'negative']
n_perc_before = 100 * len(ntweets_before)/len(tweetsBefore)

print(f'Positive tweets before {p_perc_before} %')
print(f'Neutral tweets before {100 - p_perc_before - n_perc_before} %')
print(f'Negative tweets before {n_perc_before} %')

# After
ptweets_after = dfAfter[dfAfter['sentiment'] == 'positive']
p_perc_after = 100 * len(ptweets_after)/len(tweetsAfter)

ntweets_after = dfAfter[dfAfter['sentiment'] == 'negative']
n_perc_after = 100 * len(ntweets_after)/len(tweetsAfter)

print(f'Positive tweets after {p_perc_after} %')
print(f'Neutral tweets after {100 - p_perc_after - n_perc_after} %')
print(f'Negative tweets after {n_perc_after} %')


legend = ['positive', 'negative', 'neutral']
colors = ['g', 'r', 'grey']

print ("Creating Before Chart")
slices = [p_perc_before, n_perc_before, 100 - p_perc_before - n_perc_before]

# plotting the pie chart
plt.pie(slices, labels = legend, colors = colors,
        startangle=90, shadow = True, explode = (0, 0, 0.1),
        radius = 1.2, autopct = '%1.1f%%')
 
# plotting legend
plt.legend()

# set title
plt.title("Before")
 
# showing the plot
plt.show()


print ("Creating After Chart")
slices = [p_perc_after, n_perc_after, 100 - p_perc_after - n_perc_after]

# plotting the pie chart
plt.pie(slices, labels = legend, colors = colors,
        startangle=90, shadow = True, explode = (0, 0, 0.1),
        radius = 1.2, autopct = '%1.1f%%')
 
# plotting legend
plt.legend()
 
# set title
plt.title("After")

# showing the plot
plt.show()


print ("Create T-Test for before and after sentiment mean")
a = dfBefore['score'].to_numpy()
b = dfAfter['score'].to_numpy()
print(st.ttest_ind(a=a, b=b, equal_var=True))


print ("Create T-Test for a average sentiment on number of users")
frames = [dfBefore, dfAfter]
dfAll = pd.concat(frames)
a = dfAll['score'].to_numpy()

calculatedMean = statistics.mean(dfAll['score'])
print(calculatedMean)
print(st.ttest_1samp(a=a, popmean = 0.06))


print ("Create T-Test for before and after stock price")
temp = cokeStocksBeforeAndAfterDataFrame[cokeStocksBeforeAndAfterDataFrame.index < '2021-06-14']
stockCloseBefore = temp['Adj Close'].to_numpy()

temp2 = cokeStocksBeforeAndAfterDataFrame[cokeStocksBeforeAndAfterDataFrame.index >= '2021-06-14']
stockCloseAfter = temp2['Adj Close'].to_numpy()

print(st.ttest_ind(a=stockCloseBefore, b=stockCloseAfter, equal_var=True))
















