from praw.models import MoreComments

import praw
from prawcore.exceptions import Forbidden
    
def getSubComments(comment, allComments, verbose=True):
  allComments.append(comment)
  if not hasattr(comment, "replies"):
    replies = comment.comments()
    if verbose: print("fetching (" + str(len(allComments)) + " comments fetched total)")
  else:
    replies = comment.replies
  for child in replies:
    getSubComments(child, allComments, verbose=verbose)


def getAll(r, submissionId, verbose=True):
  submission = r.submission(submissionId)
  comments = submission.comments
  commentsList = []
  for comment in comments:
    getSubComments(comment, commentsList, verbose=verbose)
  return commentsList

#res = getAll(reddit, "vs6pb2")
#res = getAll(r, "6rjwo1", verbose=False) # This won't print out progress if you want it to be silent. Default is verbose=True

def get_comments_reddit():
    
    reddit = praw.Reddit(client_id='K2ZdTSovkXwXjJHPWo00Rg', 
                         client_secret='n3-aPXJJWp957ycSn4NGFPhf90EXIQ', user_agent='testapp')

    #Quickly prints the top 25 posts in WSB: 
    
    print("")
    print('===============================================')
    print('TOP 25 HOT WALLSTREETBETS REDDIT POSTS:')
    print('===============================================')
    print("")
    hot_posts = reddit.subreddit('wallstreetbets').hot(limit=25)
    for post in hot_posts:
        print(post.title)

    subred = reddit.subreddit("wallstreetbets")
    hot_apex = subred.hot(limit=25)
    out_data = []
    
    for item in hot_apex:
        print('')
        print ('NEW POST--',item.title,'post id --', item) 
        res = getAll(reddit, item)
        
        for comment in res:
            if isinstance(comment, MoreComments):
                continue
                print(comment.body)
                out_data.append(comment.body)
            print(comment.body)
            out_data.append(comment.body)
            
    with open('reddit-data.txt', 'w', encoding="utf-8") as f:
        f.write('\n'.join(out_data))




