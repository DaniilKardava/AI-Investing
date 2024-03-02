from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import sys
import pytz


def global_exchange_status(df, markets):

    # Change to ET timezone under the assumption that data fed also has that timezone.
    df.index = df.index.tz_localize("America/New_York")
    start = min(df.index)
    end = max(df.index)

    for market in markets:
        cal = mcal.get_calendar(market)

        schedule = cal.schedule(
            start_date=start, end_date=end)

        market_open = pd.Series(0, index=df.index)

        for _, row in schedule.iterrows():
            # Mark the timestamps as open for each trading session
            market_open.loc[row.market_open:row.market_close] = 1

        df[market] = market_open

    df.index = df.index.tz_localize(None)
    return df


def class_ratio(targets):
    targets = np.argmax(targets, axis=1)
    first_class_occ = len(targets[targets == 0])
    second_class_occ = len(targets[targets == 1])
    first_freq = first_class_occ / (first_class_occ + second_class_occ)
    print("Class Frequencies: " + str(first_freq) + "/" + str(1 - first_freq))
    return np.log(first_class_occ/second_class_occ)


def binary_ohe(data, threshold_percentile):

    pct_threshold = np.percentile(
        data['Return'][data["Return"] > 0], threshold_percentile)
    print("PCT_Threshold: " + str(pct_threshold))

    if threshold_percentile == 0:
        pct_threshold = 0

    # Create a threshold classification problem
    signed_thresholded_return = (
        np.sign(data['Return'].values - pct_threshold)).reshape(-1, 1)
    signed_thresholded_return[signed_thresholded_return == 0] = -1
    encoder = OneHotEncoder(sparse=False, categories=[[-1, 1]])
    one_hot = encoder.fit_transform(signed_thresholded_return).tolist()
    data["OneHot"] = one_hot

    # Create a signed return classification column for reference
    signed_return = (
        np.sign(data['Return'].values)).reshape(-1, 1)
    signed_return[signed_return == 0] = -1
    encoder = OneHotEncoder(sparse=False, categories=[[-1, 1]])
    signed_return = encoder.fit_transform(signed_return).tolist()
    data["SignedReturn"] = signed_return

    return data


def make_multi_index(data):
    data.columns = pd.MultiIndex.from_product(
        [data.columns, ['']])
    return data


def multiclass_ohe(data, bins):

    # Check monotonic
    np_bins = np.array(bins)
    assert np.all(np_bins[1:] > np_bins[:-1]) == True

    bins.insert(0, -np.inf)
    bins.append(np.inf)
    bins = np.array(bins)
    bins = bins / 100

    data["OneHot"] = np.nan

    for i in range(len(bins) - 1):
        data["OneHot"][(data["Return"] >= bins[i]) &
                       (data["Return"] < bins[i+1])] = i
    encoder = OneHotEncoder(sparse=False, categories=[
                            np.arange(len(bins) + 1)])

    data["OneHot"] = encoder.fit_transform(
        data["OneHot"].to_numpy().reshape(-1, 1)).tolist()

    return data


def chunking(df):
    concat_df = []
    i = 1
    for chunk in df:
        print("Chunk #: " + str(i))
        i += 1
        concat_df.append(chunk)
    return pd.concat(concat_df, axis=0)


def load_filtered_news(news_dir):
    chunksize = 2000000
    news = pd.read_csv(news_dir, chunksize=chunksize, sep=',')

    news = chunking(news)
    news.set_index("Unnamed: 0", inplace=True)
    news.index.name = None

    news["Date"] = pd.to_datetime(news['Date'])

    return news


def time_features(feature_df, encoding='onehot'):

    assert encoding in ['radial', 'onehot']

    minutes = feature_df.index.minute
    hours = feature_df.index.hour
    days = feature_df.index.dayofweek
    feature_names = []

    if encoding == 'radial':
        if sum(minutes) != 0:
            minutes = np.interp(minutes, (min(minutes), max(minutes)), (0, 1))
            unique_minutes = len(set(minutes))

            feature_df["Minute_Cos"] = np.cos(
                2*np.pi / unique_minutes * minutes)
            feature_names.append("Minute_Cos")

            feature_df["Minute_Sin"] = np.sin(
                2*np.pi / unique_minutes * minutes)
            feature_names.append("Minute_Sin")

        if sum(hours) != 0:
            hours = np.interp(hours, (min(hours), max(hours)), (0, 1))
            unique_hours = len(set(hours))

            feature_df["Hour_Cos"] = np.cos(2*np.pi / unique_hours * hours)
            feature_names.append("Hour_Cos")

            feature_df["Hour_Sin"] = np.sin(2*np.pi / unique_hours * hours)
            feature_names.append("Hour_Sin")

        if sum(days) != 0:
            days = np.interp(days, (min(days), max(days)), (0, 1))
            unique_days = len(set(days))

            feature_df["Day_Cos"] = np.cos(2*np.pi / unique_days * days)
            feature_names.append("Day_Cos")

            feature_df["Day_Sin"] = np.sin(2*np.pi / unique_days * days)
            feature_names.append("Day_Sin")
    else:
        encoder = OneHotEncoder(sparse=False)
        if sum(minutes) != 0:
            data = (minutes.to_numpy()).reshape(-1, 1)
            ohe_minutes = encoder.fit_transform(data
                                                ).tolist()
            multi_index = pd.MultiIndex.from_product(
                [["Minutes"], np.arange(len(set(minutes)))])
            ohe_feature_df = pd.DataFrame(
                ohe_minutes, index=feature_df.index, columns=multi_index)

            feature_df = pd.concat(
                [feature_df, ohe_feature_df], axis=1)
            feature_names.append("Minutes")

        if sum(hours) != 0:
            data = (hours.to_numpy()).reshape(-1, 1)
            ohe_hours = encoder.fit_transform(data
                                              ).tolist()
            multi_index = pd.MultiIndex.from_product(
                [["Hours"], np.arange(len(set(hours)))])
            ohe_feature_df = pd.DataFrame(
                ohe_hours, index=feature_df.index, columns=multi_index)

            feature_df = pd.concat(
                [feature_df, ohe_feature_df], axis=1)
            feature_names.append("Hours")

        if sum(days) != 0:
            data = (days.to_numpy()).reshape(-1, 1)
            ohe_days = encoder.fit_transform(data).tolist()
            multi_index = pd.MultiIndex.from_product(
                [["Days"], np.arange(len(set(days)))])
            ohe_feature_df = pd.DataFrame(
                ohe_days, index=feature_df.index, columns=multi_index)

            feature_df = pd.concat(
                [feature_df, ohe_feature_df], axis=1)
            feature_names.append("Days")

    return feature_df, feature_names


def load_price(path):
    df = pd.read_csv(path)[::-1]

    # Load as UTC, convert to ET, drop timezone
    df["Date"] = pd.to_datetime(
        df["Date"], utc=True).dt.tz_convert("America/New_York")
    df["Date"] = df["Date"].dt.tz_localize(None)

    df.set_index("Date", inplace=True)
    df = make_multi_index(df)
    df["Return_Unshifted"] = df["Close"].pct_change()
    df["Return"] = df["Return_Unshifted"].shift(-1)

    df = df[1:-1]

    return df


def alex_crypto_sentiment(path, interval, features_df):
    # Load
    chunks = pd.read_csv(path, chunksize=2000000)
    df = chunking(chunks)
    df["Post_Time"] = pd.to_datetime(
        df["Post_Time"]).dt.tz_localize("America/New_York")
    df["Post_Time"] = df["Post_Time"].dt.tz_localize("America/New_York")

    df.set_index("Post_Time", inplace=True)

    df_eth = df[df["Ticker"] == "ETH"].drop(columns=["Ticker"])
    df_btc = df[df["Ticker"] == "BTC"].drop(columns=["Ticker"])
    df.drop(columns=["Ticker"], inplace=True)

    # Main dataframe uses distinct rows for same post if it identifies multiple Tickers. Drop dups.
    df.drop_duplicates(inplace=True)

    # Resample by interval
    df_eth = df_eth.resample(interval, closed="left", label="right").mean()
    df_btc = df_btc.resample(interval, closed="left", label="right").mean()
    df = df.resample(interval, closed="left", label="right").mean()

    df_eth.columns = ["ETH_" + col for col in df_eth.columns]
    df_btc.columns = ["BTC_" + col for col in df_btc.columns]

    joined_df = pd.concat([df, df_btc, df_eth], axis=1)

    # Offset for merge. Sentiment data for the 12:00 bar should be between 12:00 and 12:timedelta.
    joined_df.index = joined_df.index - pd.Timedelta(interval)

    # Sentiment feature names
    setiment_feature_names = list(joined_df)

    # Merge sentiment with features
    min_date = max(joined_df.index.min(), features_df.index.min())
    max_date = min(joined_df.index.max(), features_df.index.max())
    sentiment_df = joined_df[(joined_df.index >= min_date) & (
        joined_df.index <= max_date)]
    sentiment_df = make_multi_index(sentiment_df)
    features_df = features_df[(features_df.index >= min_date) & (
        features_df.index <= max_date)]

    features_df = pd.concat([features_df, sentiment_df], axis=1)
    features_df.fillna(0, inplace=True)

    return features_df, setiment_feature_names
