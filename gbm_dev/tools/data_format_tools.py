from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import sys
import pytz


def make_multi_index(df):
    '''
    Arguments
    df: dataframe with regular columns

    Convert dataframe columns to multi index

    return df
    '''

    df.columns = pd.MultiIndex.from_product(
        [df.columns, ['']])
    return df


def chunking(chunks):
    concat_df = []
    i = 1
    for chunk in chunks:
        print("Chunk #: " + str(i))
        i += 1
        concat_df.append(chunk)
    return pd.concat(concat_df, axis=0)


def time_features(df, encoding='onehot'):
    '''
    Arguments
    df: dataframe with time index
    encoding: radial or onehot

    Create columns for time features. 

    Return df
    '''
    assert encoding in ['radial', 'onehot']

    minutes = df.index.minute
    hours = df.index.hour
    days = df.index.dayofweek
    feature_names = []

    if encoding == 'radial':
        if sum(minutes) != 0:
            minutes = np.interp(minutes, (min(minutes), max(minutes)), (0, 1))
            unique_minutes = len(set(minutes))

            df["Minute_Cos"] = np.cos(
                2*np.pi / unique_minutes * minutes)
            feature_names.append("Minute_Cos")

            df["Minute_Sin"] = np.sin(
                2*np.pi / unique_minutes * minutes)
            feature_names.append("Minute_Sin")

        if sum(hours) != 0:
            hours = np.interp(hours, (min(hours), max(hours)), (0, 1))
            unique_hours = len(set(hours))

            df["Hour_Cos"] = np.cos(2*np.pi / unique_hours * hours)
            feature_names.append("Hour_Cos")

            df["Hour_Sin"] = np.sin(2*np.pi / unique_hours * hours)
            feature_names.append("Hour_Sin")

        if sum(days) != 0:
            days = np.interp(days, (min(days), max(days)), (0, 1))
            unique_days = len(set(days))

            df["Day_Cos"] = np.cos(2*np.pi / unique_days * days)
            feature_names.append("Day_Cos")

            df["Day_Sin"] = np.sin(2*np.pi / unique_days * days)
            feature_names.append("Day_Sin")
    else:
        encoder = OneHotEncoder(sparse=False)
        if sum(minutes) != 0:
            data = (minutes.to_numpy()).reshape(-1, 1)
            ohe_minutes = encoder.fit_transform(data
                                                ).tolist()
            multi_index = pd.MultiIndex.from_product(
                [["Minutes"], np.arange(len(set(minutes)))])
            ohe_df = pd.DataFrame(
                ohe_minutes, index=df.index, columns=multi_index)

            df = pd.concat(
                [df, ohe_df], axis=1)
            feature_names.append("Minutes")

        if sum(hours) != 0:
            data = (hours.to_numpy()).reshape(-1, 1)
            ohe_hours = encoder.fit_transform(data
                                              ).tolist()
            multi_index = pd.MultiIndex.from_product(
                [["Hours"], np.arange(len(set(hours)))])
            ohe_df = pd.DataFrame(
                ohe_hours, index=df.index, columns=multi_index)

            df = pd.concat(
                [df, ohe_df], axis=1)
            feature_names.append("Hours")

        if sum(days) != 0:
            data = (days.to_numpy()).reshape(-1, 1)
            ohe_days = encoder.fit_transform(data).tolist()
            multi_index = pd.MultiIndex.from_product(
                [["Days"], np.arange(len(set(days)))])
            ohe_df = pd.DataFrame(
                ohe_days, index=df.index, columns=multi_index)

            df = pd.concat(
                [df, ohe_df], axis=1)
            feature_names.append("Days")

    return df, feature_names


def load_price(path):
    '''
    Arguments
    path: string path to load from  

    Load price and set index to EST timezone.
    Make the columns all multi-index for compatability with one hot features in the future.
    Calculate a percent change column.

    Return df
    '''
    df = pd.read_csv(path)[::-1]

    # Load as UTC, convert to ET, drop timezone
    df["Date"] = pd.to_datetime(
        df["Date"], utc=True).dt.tz_convert("America/New_York")
    df["Date"] = df["Date"].dt.tz_localize(None)

    df.set_index("Date", inplace=True)
    df = make_multi_index(df)
    df["Return"] = df["Close"].pct_change()

    df = df[1:]

    return df


def save_predictions(y_true, y_pred, path):
    '''
    Arguments
    y_test: Dataframe of true values
    y_pred: Array of model predictions
    path: string path to save to

    Convert array of predictions to a dataframe. Concatenate with true values and save.
    '''
    y_pred = pd.Series(y_pred, index=y_true.index, name='Prediction')

    combined = pd.concat([y_true, y_pred], axis=1)
    combined.rename(columns={y_true.name: 'True Direction',
                             y_pred.name: 'Model Prediction'}, inplace=True)

    combined.to_csv('logs/'+path)
