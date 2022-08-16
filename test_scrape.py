from praw.models import MoreComments

import praw
from prawcore.exceptions import Forbidden

def get_comments_reddit():
    
    reddit = praw.Reddit(client_id='K2ZdTSovkXwXjJHPWo00Rg', 
                         client_secret='n3-aPXJJWp957ycSn4NGFPhf90EXIQ', user_agent='testapp')

    #Quickly prints the top 25 posts in WSB: 
    
    print("")
    print('===============================================')
    print('TOP 50 HOT WALLSTREETBETS REDDIT POSTS:')
    print('===============================================')
    print("")
    hot_posts = reddit.subreddit('wallstreetbets').hot(limit=50)
    for post in hot_posts:
        print(post.title)

    #Now we want to go through each of those 25 threads and get all of the comments. Unfortunately praw seems to be 
    #limited to pulling a max of 500 comments per threaed. I am not as firmiliar with praw or reddits API so I am not 
    #if there is a praw solution to this but I am aware of other solutions outside of praw itself but to keep things 
    #simple for now, I will explore then later. Also since we are looking at 25 threads, a max of 500 of the top comments 
    #will be more than sufficent for these exploretory purposes..

    subred = reddit.subreddit("wallstreetbets")
    hot_apex = subred.hot(limit=50)
    out_data = []
    print("")
    print('===============================================')
    print('PULLING COMMENTS FROM TOP 50 POSTS:')
    print('===============================================')
    print("")
    for item in hot_apex:
        print('')
        print ('NEW POST--',item.title,'post id --', item) 
        sub_id = reddit.submission(id=item)
        
        for top_level_comment in sub_id.comments.list():
            if isinstance(top_level_comment, MoreComments):
                continue
            print(top_level_comment.body)
            out_data.append(top_level_comment.body)           
            
    #Finally we write the data to a text file to be sent down the pipe to be ingested by the FastText AI:
            
    with open('data/reddit-data.txt', 'w', encoding="utf-8") as f:
        f.write('\n'.join(out_data))


