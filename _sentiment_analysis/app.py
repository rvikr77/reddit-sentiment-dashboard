import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for generating images

from flask import Flask, render_template, request
import praw
from textblob import TextBlob
import pandas as pd
import plotly.express as px
import plotly.io as pio
from io import BytesIO
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from collections import Counter
import seaborn as sns
from flask_sqlalchemy import SQLAlchemy
import re

# Initialize Flask app
app = Flask(__name__)

# Configure PostgreSQL database
#change username,password and db name(reddit_db)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://username:password@localhost:5432/reddit_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the database model
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String, nullable=False)
    score = db.Column(db.Integer, nullable=False)
    num_comments = db.Column(db.Integer, nullable=False)
    created_utc = db.Column(db.DateTime, nullable=False)
    sentiment = db.Column(db.Float, nullable=False)
    search_query = db.Column(db.String, nullable=False)

    def __repr__(self):
        return f'<Post {self.title}>'

# Initialize Reddit API client
#reddit api key details
reddit = praw.Reddit(
    client_id='',
    client_secret='',
    user_agent=''
)


# Path for saving the word cloud image
WORD_CLOUD_PATH = 'static/wordcloud.png'

def scrape_reddit(query, limit=10):
    subreddit = reddit.subreddit('all')  # Search in all subreddits
    posts = []
    for submission in subreddit.search(query, limit=limit):
        posts.append({
            'title': submission.title,
            'score': submission.score,
            'num_comments': submission.num_comments,
            'created_utc': pd.to_datetime(submission.created_utc, unit='s')
        })
    return pd.DataFrame(posts)


def sanitize_title(title):
    # Remove any characters that are not printable
    sanitized_title = re.sub(r'[^\x00-\x7F]+', '', title)
    return sanitized_title


def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def create_plots(df):
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    
    sentiment_over_time = px.line(df, x='created_utc', y='sentiment', title='Sentiment Over Time', labels={'created_utc': 'Date', 'sentiment': 'Sentiment Polarity'})
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
    sentiment_distribution = px.histogram(df, x='sentiment_label', title='Sentiment Distribution', labels={'sentiment_label': 'Sentiment'})
    
    sentiment_over_time_html = pio.to_html(sentiment_over_time, full_html=False)
    sentiment_distribution_html = pio.to_html(sentiment_distribution, full_html=False)
    
    return sentiment_over_time_html, sentiment_distribution_html

def create_word_cloud(df):
    text = ' '.join(df['title'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Ensure the static directory exists
    os.makedirs(os.path.dirname(WORD_CLOUD_PATH), exist_ok=True)
    
    # Save word cloud image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(WORD_CLOUD_PATH)
    plt.close()
    
    return WORD_CLOUD_PATH

def get_top_keywords(df, top_n=10):
    all_words = ' '.join(df['title'].tolist()).split()
    word_freq = Counter(all_words)
    most_common_words = word_freq.most_common(top_n)
    
    keywords_df = pd.DataFrame(most_common_words, columns=['Keyword', 'Frequency'])
    keyword_distribution = px.bar(keywords_df, x='Keyword', y='Frequency', title='Top Keywords')
    
    keyword_distribution_html = pio.to_html(keyword_distribution, full_html=False)
    return keyword_distribution_html

def create_correlation_plot(df):
    correlation = df[['score', 'num_comments', 'sentiment']].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    correlation_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    correlation_image_html = f'<img src="data:image/png;base64,{correlation_image}"/>'
    return correlation_image_html

def save_to_db(df,query):
    db.create_all()
    
    for _, row in df.iterrows():
        sanitized_title = sanitize_title(row['title'])
        post = Post(
            title=sanitized_title,
            score=row['score'],
            num_comments=row['num_comments'],
            created_utc=row['created_utc'],
            sentiment=row['sentiment'],
            search_query=query
        )
        db.session.add(post)
    db.session.commit()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        df = scrape_reddit(query)
        df['sentiment'] = df['title'].apply(analyze_sentiment)
        
        # Save to database
        save_to_db(df,query)
        
        # Create visualizations
        sentiment_over_time_html, sentiment_distribution_html = create_plots(df)
        wordcloud_path = create_word_cloud(df)
        top_keywords_html = get_top_keywords(df)
        correlation_plot_html = create_correlation_plot(df)
        
        return render_template('index.html', query=query, 
                               sentiment_over_time_html=sentiment_over_time_html, 
                               sentiment_distribution_html=sentiment_distribution_html, 
                               wordcloud_path=wordcloud_path,
                               top_keywords_html=top_keywords_html,
                               correlation_plot_html=correlation_plot_html)
    return render_template('index.html', query=None, 
                           sentiment_over_time_html=None, 
                           sentiment_distribution_html=None, 
                           wordcloud_path=None,
                           top_keywords_html=None,
                           correlation_plot_html=None)

if __name__ == '__main__':
    app.run(debug=True)
