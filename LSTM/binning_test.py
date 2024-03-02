import yfinance as yf
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import statistics
from data_format_tools import load_price
from features.feature_tools import bin_return
import pandas as pd
import sys

end = datetime(2022, 1, 1)

path = 'data/QQQ_1hour.csv'
df = load_price(path=path)

eight_bins = [-np.inf, -.75, -.25, -.1, 0, .1, .25, .75, np.inf]
features_df, indicator_names = bin_return(
    df, end, "Return_Unshifted", eight_bins, [], bin_type='custom')


for y in range(2016, 2024):
    for bin in range(0, 8):
        data = {}

        df = features_df[features_df['binned_Return_Unshifted'][bin] == 1]

        date_start = datetime(y, 1, 1)
        date_end = datetime(y+1, 1, 1)
        df = df[(df.index > date_start) & (df.index < date_end)]

        for h in range(9, 15):
            s = len(df[(df['Return'] > 0) & (df.index.hour == h)])
            t = len(df[(df.index.hour == h)])
            try:
                data[h] = [s/t, t]
            except:
                data[h] = [0, 0]

        multi_col = pd.MultiIndex.from_product([[bin], ['Accuracy', "Counts"]])
        multi_ind = pd.MultiIndex.from_product([[y], data.keys()])
        bin_df = pd.DataFrame(data=data.values(),
                              index=multi_ind, columns=multi_col)

        if bin == 0:
            bin_dfs = bin_df
        else:
            bin_dfs = pd.concat([bin_dfs, bin_df], axis=1)
    if y == 2016:
        annual_bins_dfs = bin_dfs
    else:
        annual_bins_dfs = pd.concat([annual_bins_dfs, bin_dfs], axis=0)


greens = [2016, 2017, 2019, 2021]
reds = [2018, 2020, 2021]
test = np.arange(2022, 2024)

chosen = test

for y in chosen:
    if y == chosen[0]:
        combined = annual_bins_dfs.xs(y)
    else:
        combined = combined + annual_bins_dfs.xs(y)

# Aberage the accuracy column
for b in range(8):
    combined.loc[:, (b, "Accuracy")] = combined[b]['Accuracy'] / len(chosen)

print(combined)
