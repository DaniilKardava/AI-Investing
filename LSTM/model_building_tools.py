import os
import json
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.initializers import Orthogonal, GlorotUniform, HeUniform
from keras.layers import Input, Dense, LSTM, LeakyReLU, Dropout
from keras.models import Model
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from backtest_analysis_tools import analyze_probs, calculate_return, analyze_tod
from data_format_tools import class_ratio


class lstm_network():
    def __init__(self, classifications, window_size, feature_space, learning_rate=.0007):

        self.learning_rate = learning_rate
        self.window = window_size
        self.features = feature_space

        input_layer = Input(shape=(window_size, feature_space))
        dense_layer = Dense(units=64, activation='tanh',
                            kernel_initializer=Orthogonal)(input_layer)
        dense_layer = Dropout(0.0)(dense_layer)  # Added dropout layer
        dense_layer = Dense(units=64, activation='tanh',
                            kernel_initializer=Orthogonal)(dense_layer)
        dense_layer = Dropout(0.0)(dense_layer)  # Added dropout layer
        lstm_layer = LSTM(units=256, kernel_initializer=Orthogonal,
                          recurrent_initializer=Orthogonal, recurrent_dropout=.0)(dense_layer)
        logits = Dense(classifications, activation="linear",
                       kernel_initializer=Orthogonal, )(lstm_layer)
        market_directions = keras.layers.Activation("softmax")(logits)

        self.model = Model(
            inputs=input_layer, outputs=market_directions)

        self.optimizer = keras.optimizers.legacy.Adam(
            learning_rate=self.learning_rate)

        # Compile so the model can be saved. I do not intend to use the defined loss function.
        self.model.compile(optimizer=self.optimizer,
                           loss="categorical_crossentropy")

    # Custom functions for single passes that are usually faster than the keras functions
    @tf.function
    def forward(self, one_input):
        return self.model(inputs=one_input)

    @tf.function
    def train(self, input, target):
        # Pass bootstrapped return to exclude it from gradient calculations.
        with tf.GradientTape() as tape:

            # Pass list of observations gathered in batch as a timeseries and gather outputs on each step.
            output = self.model(
                inputs=input)

            loss = keras.losses.categorical_crossentropy(
                y_pred=output, y_true=target)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))


def fit_lstm(data, features, train_start, train_end, model=None, long_only=False, window_size=30):

    train_set = data[(data.index >= train_start) & (data.index < train_end)]

    targets = np.array(list(train_set['OneHot']))
    train_set = train_set[features].to_numpy()

    if model == None:
        # Length of first row, possibly one hot encoded features.
        input_size = len(train_set[0])

        # Calculate class imbalance
        bias_initializer = class_ratio(targets=targets)
        model = lstm_network(
            classifications=2, feature_space=input_size, window_size=window_size)
    else:
        model = model
    window = model.window

    if long_only:
        classes = np.argmax(targets, axis=1)
        indices = np.where(classes == 1)
        print(indices)
        targets = targets[indices]
        print(targets)
        train_set = train_set[indices]

    # TSG uses i+1 value in targets as corresponding label. Shift data appropriately.
    targets = np.insert(targets, 0, np.zeros_like(targets[0]), axis=0)
    train_set = np.insert(train_set, len(
        train_set), np.zeros_like(train_set[0]), axis=0)

    generator = TimeseriesGenerator(
        train_set, targets, length=window, batch_size=32)

    history = model.model.fit(generator)
    loss = history.history['loss'][0]

    return model, loss


def evaluate_lstm(data, features, eval_start, eval_end, model, side="long", confidence_threshold=75, trades_by_hour={}, trade_select_hours=np.arange(25), display_plots=True):
    window = model.window
    test_set = data[(data.index >= eval_start) & (data.index < eval_end)]

    # Targets are just for generating the time series object
    targets = np.array(list(test_set['OneHot']))
    test_set_np = test_set[features].to_numpy()

    # TSG uses i+1 value in targets as corresponding label. Shift data appropriately.
    targets_adjusted = np.insert(targets, 0, np.zeros_like(targets[0]), axis=0)
    test_set_np_adjusted = np.insert(test_set_np, len(
        test_set_np), np.zeros_like(test_set_np[0]), axis=0)

    generator = TimeseriesGenerator(
        test_set_np_adjusted, targets_adjusted, length=window, batch_size=64)

    probability_outputs_mag_threshold = model.model.predict(generator)

    predicted_direction_mag_threshold = np.argmax(
        probability_outputs_mag_threshold, axis=1)
    true_mag_threshold = test_set["OneHot"][window -
                                            1:].apply(lambda x: np.argmax(x))
    true_signed = test_set["SignedReturn"][window -
                                           1:].apply(lambda x: np.argmax(x))

    # On occassion there are no classifications for a group:
    if len(set(predicted_direction_mag_threshold)) != 2:
        return None

    # Calibration:
    one_sided_model_outputs, corresponding_outcomes, _ = analyze_probs(test_set.index[window-1:],
                                                                       true_mag_threshold, true_signed, probability_outputs_mag_threshold, side=side, use_select_hours=trade_select_hours, display_plots=display_plots)
    if side == 'long':
        threshold = np.percentile(
            one_sided_model_outputs[:, 1], confidence_threshold)
    else:
        threshold = np.percentile(
            one_sided_model_outputs[:, 0], confidence_threshold)
    print("Confidence Threshold: " + str(threshold))

    # Graph returns:
    new_trades_by_hour, backtest_accuracy = calculate_return(test_set[window - 1:], probability_outputs_mag_threshold, predicted_direction_mag_threshold,
                                                             threshold=threshold, confidence_multiplier=1, side=side, use_select_hours=trade_select_hours, display_plots=display_plots)

    # Analyze performance by tod
    for key, val in new_trades_by_hour.items():
        try:
            trades_by_hour[key] = pd.concat(
                [trades_by_hour[key], val], ignore_index=True)
        except:
            trades_by_hour[key] = val

    analyze_tod(trades_by_hour, accuracies=False, returns=False, side=side)

    correct = sum(predicted_direction_mag_threshold ==
                  true_signed) / len(true_signed)
    conf = confusion_matrix(true_signed, predicted_direction_mag_threshold)

    positive_days_ratio = sum(true_signed) / len(true_signed)

    return correct, conf, positive_days_ratio, trades_by_hour, backtest_accuracy, threshold


def evaluate_models(path, features_df, feature_names, start, end, side="long", confidence_percentile=75, display_plots=True, trade_select_hours=np.arange(25), start_with=0, window_size=30, summary_name='OOS_summary'):

    if len(trade_select_hours) == 0:
        trade_select_hours = np.arange(25)

    assert side in ['long', 'short']

    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    files = sorted(files, key=os.path.getmtime)

    input_size = len(set(features_df[feature_names].columns))

    oos_summary = {"Train_Loss": [], "Total_Accuracy": [],
                   "Long_Accuracy": [], "Threshold": [], "Threshold_Accuracy": [], "Confusion": []}
    trades_by_hour = {}
    for file in files[start_with:]:
        print(file)
        model = lstm_network(
            classifications=2, feature_space=input_size, window_size=window_size)
        try:
            train_loss = file.split("$")[-2]
            model.model = keras.models.load_model(file)
        except:
            continue

        try:
            acc, conf, positive_days_ratio, trades_by_hour, backtest_accuracy, threshold = evaluate_lstm(
                features_df, feature_names, start, end, model, side, confidence_percentile, trades_by_hour, trade_select_hours, display_plots)
        except Exception as e:
            print(e)
            continue

        print("Positive Days Ratio: " + str(positive_days_ratio))
        print("Total accuracy (signed): " + str(acc))
        print('Confusion (Signed): \n{}'.format(conf))
        print("Long Accuracy: " + str(conf[1][1] / (conf[1][1] + conf[0][1])))
        print("Short Accuracy: " + str(conf[0][0] / (conf[0][0] + conf[1][0])))

        oos_summary["Train_Loss"].append(train_loss)
        oos_summary["Total_Accuracy"].append(acc)
        oos_summary["Long_Accuracy"].append(
            conf[1][1] / (conf[1][1] + conf[0][1]))
        oos_summary["Threshold"].append(threshold)
        oos_summary["Threshold_Accuracy"].append(backtest_accuracy)
        oos_summary["Confusion"].append(list(conf))

    oos_summary_df = pd.DataFrame(oos_summary)
    oos_summary_df.to_csv(path + summary_name + '.csv')


def build_models(path, features_df, feature_names, start, end, run_PCA, clustered_features, num_models=15, import_last=False, long_only=False, window_size=30):

    # Save clustered features for model context
    if run_PCA:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path + 'clustered_features.json', 'w') as file:
            json.dump(clustered_features, file)

    start_num = 0
    if import_last:
        input_size = len(set(features_df[feature_names].columns))
        model = lstm_network(
            classifications=2, feature_space=input_size, window_size=window_size)

        files = os.listdir(path)
        files = [os.path.join(path, file) for file in files]
        files = sorted(files, key=os.path.getmtime)

        for file in files[::-1]:
            try:
                model.model = keras.models.load_model(file)
                start_num = int(file.split('#')[-2]) + 1
                break
            except:
                continue

    for i in range(start_num, start_num + num_models):
        print("Model: " + str(i))
        if i == 0:
            model, loss = fit_lstm(
                features_df, feature_names, start, end, long_only=long_only, window_size=window_size)
        else:
            model, loss = fit_lstm(
                features_df, feature_names, start, end, model, long_only=long_only, window_size=window_size)

        model.model.save(path + 'n#' + str(i) +
                         "#l$" + str(loss)[:6] + "$")
