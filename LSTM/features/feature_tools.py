import math
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt


def one_hot_inputs(training_range_df, indicator_component_df, multi_index_name, feature_bins):
    quantiles = np.linspace(0, 1, feature_bins + 1)
    bins = np.quantile(training_range_df, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    binned_data = pd.cut(
        indicator_component_df, labels=False,
        bins=bins, duplicates='drop')

    encoder = OneHotEncoder(sparse=False, categories=[
        np.arange(binned_data.max() + 1)])

    indicator_component_np = encoder.fit_transform(
        binned_data.to_numpy().reshape(-1, 1))

    sub_columns = range(binned_data.max() + 1)
    multi_index = pd.MultiIndex.from_product(
        [[multi_index_name], sub_columns])

    return multi_index, indicator_component_np


def lookahead_indicator(df, indicator):
    if math.isnan(df[-1]):
        print("Skipped, Lookahead: " +
              str(indicator))
        return True
    return False


def accomodate_warmup(df):
    return df.first_valid_index()


def stationarity_test(df):
    df = df.dropna()
    p_value = adfuller(df, maxlag=5)[1]
    if p_value > .05:
        df = df.pct_change()
    return df


def compare_lengths(df, sent_df):
    return len(df.index) == len(sent_df.index)


def custom_interp(df_limits, df):
    data_min = min(df_limits)
    data_max = max(df_limits)
    if data_min < 0:
        lower_limit = -1
    else:
        lower_limit = 0
    if data_max > 0:
        upper_limit = 1
    else:
        upper_limit = 0
    df_interp = np.interp(
        df, (data_min, data_max), (lower_limit, upper_limit))
    return df_interp


def median_binning(training_range_df, df, feature_bins):
    '''
    Calculate quantiles based on raw train data, apply to all raw data. Then normalize data, apply bins by index, and calculate 
    medians. Replace bin number with medians. 
    Parameters:
    training_range_df (pd.DataFrame): Dataframe containing feature values excluding test data.
    df (pd.DataFrame): Dataframe containing feature values including both train and test
    feature_bins (int): number of bins
    Returns:
    np.array: normalized feature values as median of corresponding bin.
    '''
    quantiles = np.linspace(0, 1, feature_bins + 1)
    bins = np.quantile(training_range_df, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    binned_data = pd.cut(
        df, bins=bins, labels=False, duplicates='drop')

    normalized_data_np = custom_interp(training_range_df, df)
    df = pd.DataFrame(data=normalized_data_np, index=binned_data.index)

    bin_medians = df.groupby(binned_data).median(1)

    for i, med in bin_medians.iterrows():
        med = med[0]
        binned_data[binned_data == i] = med

    return binned_data


def mag_returns(df, name, feature_names):
    '''
    Given dataframe, column name, and feature list, return dataframe with absolute value of that column and new feature list.
    '''
    new_col_name = 'abs_' + name
    df[new_col_name] = np.abs(df[name])
    feature_names.append(new_col_name)

    return df, feature_names


def sign_return(df, name, feature_names):
    '''
    Given dataframe, column name, and feature list, return dataframe with sign of value of that column and new feature list.
    '''
    new_col_name = 'sign_' + name
    df[new_col_name] = np.sign(df[name])
    feature_names.append(new_col_name)

    return df, feature_names


def bin_return(df, training_end, name, quantiles, feature_names, bin_type):
    '''
    Given dataframe, training range end date, column name, quartiles, and feature list, return dataframe with binned and one hot encoded column.
    '''

    assert bin_type in ['custom', 'quartile']

    if bin_type == 'quartile':
        train_df = df[df.index < training_end]
        bins = np.quantile(train_df['Return'], quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
    else:
        bins = np.array(quantiles) / 100

    new_col_name = 'binned_' + name
    binned_data = pd.cut(
        df[name], bins=bins, labels=False, duplicates='drop')

    encoder = OneHotEncoder(sparse=False)

    ohe_data = encoder.fit_transform(
        binned_data.to_numpy().reshape(-1, 1))

    sub_columns = range(binned_data.max() + 1)
    multi_index = pd.MultiIndex.from_product(
        [[new_col_name], sub_columns])

    ohe_df = pd.DataFrame(ohe_data, index=df.index, columns=multi_index)

    df = pd.concat([df, ohe_df], axis=1)

    feature_names.append(new_col_name)

    return df, feature_names


def volume_indicator(df, name, feature_names):
    '''
    Given df and name of volume column.
    '''
    window = 30
    df["Average_Volume"] = df['Volume'].rolling(
        window=window).mean()
    df[name] = df["Volume"] / \
        df["Average_Volume"]
    feature_names.append(name)
    df = df[30:]
    return df, feature_names
