import pandas as pd
from datetime import datetime

df = pd.read_csv('data/SPY_15Min_2008-Sep2023 (1).csv')
print(df)

df["Time"] = pd.to_datetime(df["Time"]).dt.tz_localize(
    'America/New_York', ambiguous='infer')
df.drop(columns=['Change', '%Chg'], inplace=True)

df.set_index('Time', inplace=True)
df.index.name = "Date"

df = df.loc[df.index.date > datetime(2014, 1, 1).date()]
print(df)

df.to_csv('data/SPY_15min.csv')
