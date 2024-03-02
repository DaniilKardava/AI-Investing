from tools.data_format_tools import load_price, make_multi_index, save_predictions
from tools.feature_tools import bin_return, rf_anchored_walkforward, rf_rolling_walkforward, rolling_accuracy, anchored_accuracy, analyze_features, name_bin_features
import numpy as np
from datetime import datetime

# Data load
path = 'C:/Users/danik/QT.Evan/gbm_dev/data/'
df = load_price(path+'SPY_15min.csv')

# Calculate y
period_start = df.loc[(df.index.hour == 11) & (df.index.minute == 0)]
period_end = df.loc[(df.index.hour == 15) & (df.index.minute == 0)]

period_start.index = period_start.index.date
period_end.index = period_end.index.date

y = (period_end['Open'] - period_start['Open']) / period_start['Open'] > 0
y = y.astype(int)
y.name = 'Target'

# Calculate X
period_start = df.loc[(df.index.hour == 9) & (df.index.minute == 30)]
period_end = df.loc[(df.index.hour == 11) & (df.index.minute == 0)]

period_start.index = period_start.index.date
period_end.index = period_end.index.date

X = (period_end['Open'] - period_start['Open']) / period_start['Open']
X.name = 'Returns'

# Drop NaN rows and save intersection of the series
X.dropna(inplace=True)
y.dropna(inplace=True)
intersection = X.index.intersection(y.index)
X = X.loc[intersection]
y = y.loc[intersection]


# Convert to dataframe for compatability with imported binning function. Bin X.
X = X.to_frame()
X = make_multi_index(X)
bins = [-np.inf, -.8, -.39, -.05, 0, .05, .39, .8, np.inf]
X, feature_name = bin_return(
    X, "Returns", bins)
X = X[feature_name]

X = name_bin_features(X, X.columns, bins, 'r')


print(f'Features: \n{X}')
print(f'Target: \n{y}')


# Parameters for model
params = {
}

# Implement rolling walkforward

y_true, y_pred, feature_gains = rf_rolling_walkforward(
    X, y, params, lookback=200, lookforward=1, correct_imbalance=False)

analyze_features(feature_gains, importance_type='Gains',
                 train_method='Rolling')


save_predictions(y_true, y_pred, 'Rolling.csv')

y_pred = [1 if i > .5 else 0 for i in y_pred]

comparison = y_pred == y_true

print(f'Accuracy: {comparison.sum() / comparison.size}')

# Implement anchored walkforward

y_true, y_pred, feature_gains = rf_anchored_walkforward(
    X, y, params, lookback=200, lookforward=1, correct_imbalance=False)

analyze_features(feature_gains, importance_type='Gains',
                 train_method='Anchored')

save_predictions(y_true, y_pred, 'Anchored.csv')

y_pred = [1 if i > .5 else 0 for i in y_pred]

comparison = y_pred == y_true


print(f'Accuracy: {comparison.sum() / comparison.size}')
