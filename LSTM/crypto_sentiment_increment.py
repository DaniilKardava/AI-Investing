'''Increment the hours of crypto sentiment for Evan's correlation strategy.'''
import pandas as pd
from datetime import datetime, timedelta
from data_format_tools import chunking

path = 'data/hourly_sentiment.csv'
df = pd.read_csv(path, chunksize=1000000)
df = chunking(df)

df['Post_Time'] = pd.to_datetime(df['Post_Time'], utc=True)
increment = timedelta(hours=1)

print(df)

df['Post_Time'] = df['Post_Time'] - increment

print(df)

df.to_csv('data/hourly_sentiment_decremented.csv')
