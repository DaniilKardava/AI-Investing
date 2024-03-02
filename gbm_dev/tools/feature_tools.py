import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import itertools


def rf_rolling_walkforward(X, y, params, lookback, lookforward, correct_imbalance, subset=None):
    '''
    Arguments
    X: Features dataframe
    y: target dataframe
    params: rf params dictionary
    lookback: train length for each prediction
    lookforward: predict length for each model
    subset: day index subset to train and evaluate on

    Train on lookback length and predict next period on a rolling basis. Keep track of relative feature weights and gains

    return true, predictions, relative feature gains, relative feature weights
    '''

    step = lookback
    data_size = y.size

    overall_gains = {i: {'gain': [], 'std': []} for i in X.columns}

    predictions = []
    true = []
    prediction_dates = []

    print(f'Data Size: {data_size}')

    while step < data_size:

        # Check if the day is part of the random sample
        if subset is not None:
            if X.index[step] not in subset:
                step += lookforward
                continue

        if step % 100 == 0:
            print(f'Current Step: {step}')

        X_train, X_test = X.iloc[step -
                                 lookback:step], X.iloc[step:step + lookforward]
        y_train, y_test = y.iloc[step -
                                 lookback:step], y.iloc[step:step + lookforward]

        # Pass params here later
        if correct_imbalance:
            clf = RandomForestClassifier(
                class_weight='balanced', n_jobs=-1)
        else:
            clf = RandomForestClassifier(n_jobs=-1)

        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)

        # Format two class predictions
        formatted_pred = []
        for i, prediction in enumerate(y_pred):
            pred_class = np.argmax(prediction)
            if pred_class == 0:
                formatted_pred.append(1 - np.max(prediction))
            else:
                formatted_pred.append(np.max(prediction))
        y_pred = formatted_pred

        predictions.extend(y_pred)

        # Record true outcomes and dates to rebuild dataframe
        true.extend(y[step:step+lookforward])
        prediction_dates.extend(y.index[step:step+lookforward])

        # Record the feature importance
        importances = clf.feature_importances_
        importances_std = np.std(
            [tree.feature_importances_ for tree in clf.estimators_], axis=0)

        # Record relative gains:
        total_gain = sum(importances)

        for i, key in enumerate(overall_gains.keys()):

            gain = importances[i] / total_gain
            std = importances_std[i] / total_gain
            overall_gains[key]['gain'].append(gain)
            overall_gains[key]['std'].append(std)
        step += lookforward

    true = pd.Series(data=true, index=prediction_dates, name='True Direction')

    return true, predictions, overall_gains


def rf_anchored_walkforward(X, y, params, lookback, lookforward, correct_imbalance, subset=None):
    '''
    Arguments
    X: Features dataframe
    y: target dataframe
    params: rf dictionary
    lookback: initial length for first prediction
    lookforward: predict length for each model
    subset: day index subset to train and evaluate on

    Train on anchored lookback. Keep track of relative feature weights and gains

    return true, predictions, relative feature gains, relative feature weights
    '''

    step = lookback
    data_size = y.size

    overall_gains = {i: {'gain': [], 'std': []} for i in X.columns}

    predictions = []
    true = []
    prediction_dates = []

    print(f'Data Size: {data_size}')

    while step < data_size:

        # Check if the day is part of the random sample
        if subset is not None:
            if X.index[step] not in subset:
                step += lookforward
                continue

        if step % 100 == 0:
            print(f'Current Step: {step}')

        X_train, X_test = X.iloc[:step], X.iloc[step:step + lookforward]
        y_train, y_test = y.iloc[:step], y.iloc[step:step + lookforward]

        # Pass params here later
        if correct_imbalance:
            clf = RandomForestClassifier(
                class_weight='balanced', n_jobs=-1)
        else:
            clf = RandomForestClassifier(n_jobs=-1)

        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)

        # Format two class predictions
        formatted_pred = []
        for i, prediction in enumerate(y_pred):
            pred_class = np.argmax(prediction)
            if pred_class == 0:
                formatted_pred.append(1 - np.max(prediction))
            else:
                formatted_pred.append(np.max(prediction))
        y_pred = formatted_pred

        predictions.extend(y_pred)

        # Record true outcomes and dates to rebuild dataframe
        true.extend(y[step:step+lookforward])
        prediction_dates.extend(y.index[step:step+lookforward])

        # Record the feature importance
        importances = clf.feature_importances_
        importances_std = np.std(
            [tree.feature_importances_ for tree in clf.estimators_], axis=0)

        # Record relative gains:
        total_gain = sum(importances)

        for i, key in enumerate(overall_gains.keys()):

            gain = importances[i] / total_gain
            std = importances_std[i] / total_gain
            overall_gains[key]['gain'].append(gain)
            overall_gains[key]['std'].append(std)

        step += lookforward

    true = pd.Series(data=true, index=prediction_dates, name='True Direction')

    return true, predictions, overall_gains


def xgb_rolling_walkforward(X, y, params, rounds, lookback, lookforward, correct_imbalance, subset=None):
    '''
    Arguments
    X: Features dataframe
    y: target dataframe
    params: xgb dictionary
    rounds: number of trees
    lookback: train length for each prediction
    subset: day index subset to train and evaluate on

    Train on lookback length and predict next period on a rolling basis. Keep track of relative feature weights and gains

    return true, predictions, relative feature gains, relative feature weights
    '''

    step = lookback
    data_size = y.size

    # Add std to conform with the importance plot I generate later
    overall_weights = {i: {'gain': [], 'std': []} for i in X.columns}
    overall_gains = {i: {'gain': [], 'std': []} for i in X.columns}

    predictions = []
    true = []
    prediction_dates = []

    print(f'Data Size: {data_size}')

    while step < data_size:

        # Check if the day is part of the random sample
        if subset is not None:
            if X.index[step] not in subset:
                step += lookforward
                continue

        X_train, X_test = X.iloc[step -
                                 lookback:step], X.iloc[step:step + lookforward]
        y_train, y_test = y.iloc[step -
                                 lookback:step], y.iloc[step:step + lookforward]

        if correct_imbalance:
            class_imbalance = (y_train == 0).sum() / (y_train == 1).sum()
            sample_weights = [
                1 if i == 0 else class_imbalance for i in y_train]
        else:
            sample_weights = [1] * y_train.size

        dtrain = xgb.DMatrix(X_train, label=y_train,
                             weight=sample_weights)
        dtest = xgb.DMatrix(X_test, label=y_test)

        bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=rounds)

        y_pred = bst.predict(dtest)
        predictions.extend(y_pred)

        # Record true outcomes and dates to rebuild dataframe
        true.extend(y[step:step+lookforward])
        prediction_dates.extend(y.index[step:step+lookforward])

        # Record the feature importance
        model_gain = bst.get_score(importance_type='gain')
        model_weight = bst.get_score(importance_type='weight')

        # Record relative weights:
        total_weight = sum(model_weight.values())

        for key in overall_weights.keys():

            if key in list(model_weight.keys()):

                val = model_weight[key] / total_weight
                overall_weights[key]['gain'].append(val)

            else:

                overall_weights[key]['gain'].append(0)

        # Record relative gains:
        total_gain = sum(model_gain.values())

        for key in overall_gains.keys():

            if key in list(model_gain.keys()):

                val = model_gain[key] / total_gain
                overall_gains[key]['gain'].append(val)

            else:

                overall_gains[key]['gain'].append(0)

        step += lookforward

    true = pd.Series(data=true, index=prediction_dates, name='True Direction')

    return true, predictions, overall_gains, overall_weights


def xgb_anchored_walkforward(X, y, params, rounds, lookback, lookforward, correct_imbalance, subset=None):
    '''
    Arguments
    X: Features dataframe
    y: target dataframe
    params: xgb dictionary
    rounds: number of trees
    lookback: initial length for first prediction
    subset: day index subset to train and evaluate on

    Train on anchored lookback. Keep track of relative feature weights and gains

    return true, predictions, relative feature gains, relative feature weights
    '''

    step = lookback
    data_size = y.size

    # Add std to conform with the importance plot I generate later
    overall_weights = {i: {'gain': [], 'std': []} for i in X.columns}
    overall_gains = {i: {'gain': [], 'std': []} for i in X.columns}

    predictions = []
    true = []
    prediction_dates = []

    print(f'Data Size: {data_size}')

    while step < data_size:

        # Check if the day is part of the random sample
        if subset is not None:
            if X.index[step] not in subset:
                step += lookforward
                continue

        X_train, X_test = X.iloc[:step], X.iloc[step:step + lookforward]
        y_train, y_test = y.iloc[:step], y.iloc[step:step + lookforward]

        if correct_imbalance:
            class_imbalance = (y_train == 0).sum() / (y_train == 1).sum()
            sample_weights = [
                1 if i == 0 else class_imbalance for i in y_train]
        else:
            sample_weights = [1] * y_train.size

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dtest = xgb.DMatrix(X_test, label=y_test)

        bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=rounds)

        y_pred = bst.predict(dtest)
        predictions.extend(y_pred)

        # Record true outcomes and dates to rebuild dataframe
        true.extend(y[step:step+lookforward])
        prediction_dates.extend(y.index[step:step+lookforward])

        # Record the feature importance
        model_gain = bst.get_score(importance_type='gain')
        model_weight = bst.get_score(importance_type='weight')

        # Record relative weights:
        total_weight = sum(model_weight.values())

        for key in overall_weights.keys():

            if key in list(model_weight.keys()):

                val = model_weight[key] / total_weight
                overall_weights[key]['gain'].append(val)

            else:

                overall_weights[key]['gain'].append(0)

        # Record relative gains:
        total_gain = sum(model_gain.values())

        for key in overall_gains.keys():

            if key in list(model_gain.keys()):

                val = model_gain[key] / total_gain
                overall_gains[key]['gain'].append(val)

            else:

                overall_gains[key]['gain'].append(0)

        step += lookforward

    true = pd.Series(data=true, index=prediction_dates, name='True Direction')

    return true, predictions, overall_gains, overall_weights


def bin_return(df, name, bins):
    '''
    ARguments
    df: dataframe with price data
    name: name of the column to bin
    bins: custom bin edges

    Apply custom bin edges on padas column.

    Return binned data and feature column name
    '''

    bins = np.array(bins) / 100

    feature_name = 'binned_' + name
    binned_data = pd.cut(
        df[name], bins=bins, labels=False, duplicates='drop')

    encoder = OneHotEncoder(sparse=False)

    ohe_data = encoder.fit_transform(
        binned_data.to_numpy().reshape(-1, 1))

    sub_columns = range(binned_data.max() + 1)
    multi_index = pd.MultiIndex.from_product(
        [[feature_name], sub_columns])

    ohe_df = pd.DataFrame(ohe_data, index=df.index, columns=multi_index)

    df = pd.concat([df, ohe_df], axis=1)

    return df, feature_name


def rolling_accuracy(y_true, y_pred, period):
    '''
    Arguments
    y_true: dataframe of true class
    y_pred: array of predicted class
    period: rolling window period

    Calculate rolling accuracies.

    return rolling accuracies
    '''

    # Convert prediction to series
    y_pred = pd.Series(y_pred, index=y_true.index) if not isinstance(
        y_pred, pd.Series) else y_pred

    rolling = (y_true == y_pred).rolling(window=period).mean()

    plt.plot(rolling)
    plt.title(f'Rolling Accuracy: Window {period}')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.show()


def anchored_accuracy(y_true, y_pred):
    '''
    Arguments
    y_true: dataframe of true class
    y_pred: array of predicted class
    period: rolling window period

    Calculate anchored accuracy.

    return rolling accuracies
    '''

    # Convert prediction to series
    y_pred = pd.Series(y_pred, index=y_true.index) if not isinstance(
        y_pred, pd.Series) else y_pred

    anchored = (y_true == y_pred).expanding().mean()

    plt.plot(anchored)
    plt.title('Anchored Accuracy')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.show()


def analyze_features(features, importance_type, train_method):
    '''
    Arguments
    features: a dictionary containing arrays of feature gains or weights from each of the models during the rolling train process
    type: label, gain or weight

    Plot cumulative gain or weight of each feature as a function of model number / time. 
    '''

    for feature, importance in features.items():

        gain = importance['gain']
        std = np.array(importance['std'])

        plt.plot(np.cumsum(gain), label=feature)

        if len(std) != 0:
            plt.fill_between(x=np.arange(len(gain)),
                             y1=np.cumsum(gain) - std,
                             y2=np.cumsum(gain) + std,
                             alpha=.1)

    plt.legend()
    plt.xlabel('Model Number')
    plt.ylabel(f'Cumulative {importance_type}')
    plt.title(f'{train_method} Cumulative {importance_type} per Feature')
    plt.show()


def name_bin_features(X, columns, bins, prefix):
    '''
    Arguments
    X: feature df
    columns: columns to rename
    bins: bin limits to rename to
    suffix: some unique identifier
    '''

    mapper = {}
    for old_col, edges in zip(columns, zip(bins[:-1], bins[1:])):

        l_edge = edges[0]
        r_edge = edges[1]

        feature_name = f'{prefix}_{l_edge}:{r_edge}'

        mapper[old_col] = feature_name

    X.rename(columns=mapper, inplace=True)

    return X


def tuning(X, y, tester, params):
    '''
    Arguments
    X: feature data
    y: labels
    tester: backtest function definition
    params: dictionary with parameter keys and list of values to search over
    correct_imbalance: boolean option to correct class imbalance
    '''

    state_space = itertools.product(*[val for key, val in params.items()])

    total_accuracy_dict = {}
    long_accuracy_dict = {}
    confident_accuracy_dict = {}

    subset = day_subset(X)

    for state in state_space:
        print(datetime.now())

        param_instance = {key: val for key, val in zip(params.keys(), state)}
        print(param_instance)

        boost_rounds = param_instance['rounds']
        correct_imbalance = param_instance['correct_imbalance']
        param_instance.pop('rounds', None)
        param_instance.pop('correct_imbalance', None)

        # Hardcoding lookback and forward for now
        y_true, y_pred, *other = tester(X=X, y=y, params=param_instance, rounds=boost_rounds, lookback=200,
                                        lookforward=1, correct_imbalance=correct_imbalance, subset=subset)

        # Analyze the confidence of long trades only
        long_only = [i for i in y_pred if i > .5]
        threshold = np.quantile(long_only, .75)

        high_long_confidence_indices = np.where(y_pred >= threshold)

        high_confidence_accuracy = sum(
            y_true.iloc[high_long_confidence_indices] == 1) / len(y_true.iloc[high_long_confidence_indices])
        confident_accuracy_dict[state] = high_confidence_accuracy

        # Analyze total accuracy
        class_prediction = [1 if i > .5 else 0 for i in y_pred]
        total_accuracy_dict[state] = sum(
            class_prediction == y_true) / len(y_true)

        # Analyze long only accuracy
        long_class_prediction_indices = np.where(
            class_prediction == np.int8(1))

        long_accuracy = sum(y_true.iloc[long_class_prediction_indices]
                            == 1) / len(y_true.iloc[long_class_prediction_indices])
        long_accuracy_dict[state] = long_accuracy

    total_accuracy_dict = dict(
        sorted(total_accuracy_dict.items(), key=lambda item: item[1]))
    confident_accuracy_dict = dict(
        sorted(confident_accuracy_dict.items(), key=lambda item: item[1]))
    long_accuracy_dict = dict(
        sorted(long_accuracy_dict.items(), key=lambda item: item[1]))

    return total_accuracy_dict, long_accuracy_dict, confident_accuracy_dict


def day_subset(X):
    '''
    Arguments
    X: train dataframe with datetime index

    Systematically select a subset of days to speed up hyperparameter search

    return datetime index
    '''

    dates = pd.to_datetime(X.index)

    weeks_of_year = dates.isocalendar().week
    years = dates.year

    df = pd.DataFrame(data={'Weeks': weeks_of_year,
                      'Year': years}, index=dates)

    df = df.groupby(['Year', 'Weeks'])

    df = df.apply(lambda x: x.sample(1))

    dates = df.index.get_level_values(2)
    dates = dates.sort_values()
    dates = dates.date

    return dates
