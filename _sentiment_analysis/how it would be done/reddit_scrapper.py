import praw
import pandas as pd

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id='',
    client_secret='',
    user_agent=''
)

def scrape_reddit(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        posts.append({
            'title': post.title,
            'score': post.score,
            'num_comments': post.num_comments,
            'created_utc': post.created_utc
        })
    return pd.DataFrame(posts)

# Example usage
df = scrape_reddit('python', limit=10)
df.to_csv('reddit_data.csv', index=False)
