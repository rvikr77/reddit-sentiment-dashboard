from textblob import TextBlob
import pandas as pd

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Load data
df = pd.read_csv('reddit_data.csv')

# Analyze sentiment
df['sentiment'] = df['title'].apply(analyze_sentiment)

# Save with sentiment
df.to_csv('reddit_data_with_sentiment.csv', index=False)
