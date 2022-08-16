#from keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText
import numpy as np
#import matplotlib.pyplot as plt
import nltk
nltk.download('omw-1.4')
#from string import punctuation
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
#from nltk import WordPunctTokenizer

#import fasttext
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

import re
#from nltk.stem import WordNetLemmatizer
import reddit_scrape
#import test_scrape
#from sklearn.decomposition import PCA

#The next step is to clean our text data by removing punctuations and numbers. We will also convert the data 
#into the lower case. The words in our data will be lemmatized to their root form. Furthermore, the stop words 
#and the words with the length less than 3 will be removed from the corpus.

stemmer = WordNetLemmatizer()

def preprocess_text(document):
       
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 1]

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text
    
def main():
    
    with open('reddit-data.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    text = sent_tokenize(text)
    final_corpus = [preprocess_text(sentence) for sentence in text if sentence.strip() !='']
    word_punctuation_tokenizer = nltk.WordPunctTokenizer()
    word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]

    #We have preprocessed our corpus. Now is the time to create word representations using FastText. Let's 
    #first define the parameters for our FastText model:
        
    embedding_size = 60
    window_size = 60
    min_word = 5
    down_sampling = 1e-2

    #now we can define the model:
 
    ft_model = FastText(word_tokenized_corpus,
                          vector_size=embedding_size,
                          window=window_size,
                          min_count=min_word,
                          sample=down_sampling,
                          sg=1,
                          epochs=100)

    #and now we can create a list of tickers, names, phrases etc. that we are interested in:
        
    master_tickers = ['tsla', 'amzn', 'qqq', 'spy', 'aapl', 'gme', 'meta', 'oil', 'amc', 'msft', 
                      'amd', 'nvda', 'msft', 'dis', 'hood', 'coin', 'baba', 'snap', 'sofi', 
                      'f', 'bb', 'twtr', 'bbby']
        
    semantically_similar_words = {words: [item[0] for item in ft_model.wv.most_similar([words], topn=20)]
                      for words in master_tickers}
    
    print('')
    print('===============================================')
    print('NLP MODEL OUTPUT:')
    print('===============================================')
    print("")
    
    for k,v in semantically_similar_words.items():
        print(k+":"+str(v))
    
    print(' ')       
    all_similar_words = sum([[k] + v for k, v in semantically_similar_words.items()], [])
    print(all_similar_words)
    
    #Here we look at the cosine similarity between some tickers and common bearish/bullish terms:
    
    avg_bull_out = []
    for l in master_tickers:
        print(" ")
        print(f"Cosine Similarity between {l} and bullsih terms:")
        bull_words = ['bull', 'bullish', 'green', 'moon', 'buy', 'calls', 'long', 'up', 'rally', 
                      'pump']
        bull_out = []
        for i in bull_words:
            cs = ft_model.wv.similarity(w1=f"{l}", w2=f"{i}")
            print(f"The Cosine Similarity between {l} and " + i + " is:")
            print(cs)
            bull_out.append(cs)
        print(f"The average Cosine Similarirty between {l} and select bullish terms is:")
        avg = np.average(bull_out)
        avg_bull_out.append(avg)
        print(avg)
    
    avg_bear_out = []
    for k in master_tickers:
        print("")
        print(f"Cosine Similarity between {k} and bearish terms:")
        bear_words = ['bear', 'bearish', 'red', 'crash', 'puts', 'sell', 'short', 'down', 'tank', 
                      'dump']
        bear_out = []
        for j in bear_words:
            cs2 = ft_model.wv.similarity(w1=f"{k}", w2=f"{j}")
            print(f"The Cosine Similarity between {k} and " + j + " is:")
            print(cs2)
            bear_out.append(cs2)
        print(f"The average Cosine Similarirty between {k} and select bearish terms is:")
        avg2 = np.average(bear_out)
        avg_bear_out.append(avg2)
        print(avg2)
        
    print('')
    print('======================================================================================')
    print('SENTIMENT BASED ON COSINE SIMILARITY OF TICKERS AND BULLISH/BEARISH TERMS FROM WSB:')
    print('(Based on comments at time program is ran!)')
    print('======================================================================================')
    print("")
    
    for x, y, z in zip(avg_bull_out, avg_bear_out, master_tickers):
        if x >= y:
            print(f"{z} has a higher average bullish term sentiment, with bullish term average CS of {x} compared to a bearish term average CS of {y} --> {z} BULLISH")
        else:
            print(f"{z} has a higher average bearish term sentiment, with bearish term average CS of {y} compared to a bullish term average CS of {x} --> {z} BEARISH")
    
    bull_avg = np.average(avg_bull_out)
    bear_avg = np.average(avg_bear_out)
    print("")
    if bull_avg >= bear_avg:
        print(f"Average WSB sentiment based on comments for the top 25 hot posts is BULLISH with an average bullish CS of {bull_avg}, and an average bearish CS of {bear_avg}")
    else:
        print(f"Average WSB sentiment based on comments for the top 25 hot posts is BEARISH with an average bearish CS of {bear_avg}, and an average bearish CS of {bull_avg}")
                
if __name__ =='__main__':
    
    reddit_scrape.get_comments_reddit()
    main()
    
