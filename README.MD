**Wallstreetbets NLP model looks at Cosine Similarity between some common tickers and a few bullish/bearish words!**

**Summary**

As a weekend project, I decided to look at some basic sentiment on the popular Subreddit WSB. I wanted to keep things very simple, so this just looks at the cosine similarity between a couple dozen popular tickers and short lists of bullish/bearish terms. 

**Quick Start**

-Clone the repo
-Navigate to where you cloned the repo
-Make sure you have Docker Desktop installed!!
-run:
`docker-compose up`

Note: WSB can be a very active forum, so depending on the time of day that you run this, it can take a while to pull comments (over an hour).
WARNING: Verbose is turned on when scraping comments from Reddit, so the output will track and print out some of the comments as it pulls them. So be aware, foul language may appear!

**Code Description**

First we have reddit_scrape.py, which pulls the top 25 posts on WSB at the time, and then iterates though all the comments and collects the text. 

I am use the praw library and API for Reddit. I believe the long pull time is coming from iterating though all the comments that I want. Initially I had used praw to pull all the comments at once. This was much faster but I was being limited to either 500 or 1000 comments per thread. Since a few of the top posts can easily have more than 10K comments, I decided to go with the slower option for complete data. 

I have worked with praw a few times before but since this was just a fun weekend project, I did not feel like digging to much into it. I will leave that to future work!

Next we have nlp_model.py, which looks at the text data we collected from WSB. 

The first thing the main part does is pull in the data and quickly process it for special characters, single characters, multiple spaces, converting to lowercase, etc. and lemmatization, into a final corpus. The FastText model is then defined and all the words in out corpus are vectorized.  

To keep things simple, I am looking just at a master list of a handful of tickers: 'tsla', 'amzn', 'qqq', 'spy', 'aapl', 'gme', 'meta', 'oil', 'amc', 'msft', 'amd', 'nvda', 'msft', 'dis', 'hood', 'coin', 'baba', 'snap', 'sofi', 'f', 'bb', 'twtr', 'bbby'.

I then look at the cosine similarity between these tickers and a few bullish terms: 'bull', 'bullish', 'green', 'moon', 'buy', 'calls', 'long', 'up', 'rally', 'pump', and bearish terms: 'bear', 'bearish', 'red', 'crash', 'puts', 'sell', 'short', 'down', 'tank', 'dump'.

**Output**

The model output is basic. First, for each ticker, it displays the average CS per ticker for bullish terms, as well as the CS for each term for that particular ticker. Then it displays the average CS per ticker for bearish terms, as well as the CS for each term for that particular ticker. 

At the end of the preliminary output it displays these averages-per-ticker again, all together, as well as the average for all of the tickers. 

**Future Work**

I would like to look at a master list of all, or most of the common tickers today in order to get a broader view. Or maybe just look at the top 50 or 100 most mentioned tickers.

Adding some sort of visualizations would be nice as well. 

I want to speed up the data-pulling. Will need to look deeper into praw. 

This was just a weekend project I did for fun recently and I would like to do something more robust than just looking at cosine similarity averages etc.
