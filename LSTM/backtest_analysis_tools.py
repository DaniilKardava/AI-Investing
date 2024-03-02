from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
import pandas as pd
from datetime import datetime
from scipy.stats import chi2_contingency, f_oneway, linregress
import statistics


def analyze_probs(time_index, true_mag, true_signed, pred, side="both", use_select_hours=np.arange(25), display_plots=True):

    assert side in ["long", "short", "both"]
    true_mag = np.array(true_mag)
    true_signed = np.array(true_signed)

    if side == "long":

        # Filter long trades
        pred_directions = np.argmax(pred, axis=1)
        long_prediction_indices = np.where(pred_directions == 1)
        selected_hours = np.array(np.where(time_index.hour.isin(
            use_select_hours))) - 1
        index_filter = np.intersect1d(long_prediction_indices, selected_hours)

        # Apply filter on predictions and outcomes
        pred = pred[index_filter]
        corresponding_outcomes_mag = true_mag[index_filter]
        corresponding_outcomes_signed = true_signed[index_filter]

        # Calculate calibration curve
        prob_true_mag, prob_pred_mag = calibration_curve(
            corresponding_outcomes_mag, np.max(pred, axis=1), pos_label=1, n_bins=5, strategy='quantile')
        prob_true_signed, prob_pred_signed = calibration_curve(
            corresponding_outcomes_signed, np.max(pred, axis=1), pos_label=1,  n_bins=5, strategy='quantile')

    elif side == "short":
        # Filter short trades
        pred_directions = np.argmax(pred, axis=1)
        short_prediction_indices = np.where(pred_directions == 0)
        selected_hours = np.array(np.where(time_index.hour.isin(
            use_select_hours))) - 1
        index_filter = np.intersect1d(short_prediction_indices, selected_hours)

        # Apply filter on predictions and outcomes
        pred = pred[index_filter]
        corresponding_outcomes_mag = true_mag[index_filter]
        corresponding_outcomes_signed = true_signed[index_filter]

        # Calculate calibration curve
        prob_true_mag, prob_pred_mag = calibration_curve(
            corresponding_outcomes_mag, np.max(pred, axis=1), pos_label=0, n_bins=5, strategy='quantile')
        prob_true_signed, prob_pred_signed = calibration_curve(
            corresponding_outcomes_signed, np.max(pred, axis=1), pos_label=0, n_bins=5, strategy='quantile')
    else:
        prob_true, prob_pred = calibration_curve(
            true_mag, np.max(pred, axis=1), n_bins=5, strategy='quantile')

    if display_plots:
        plt.scatter(prob_pred_mag, prob_true_mag,
                    label="Thresholded Classification")
        plt.scatter(prob_pred_signed, prob_true_signed,
                    label="Signed Classification")
        plt.legend()
        plt.title("Calibration")
        plt.xlabel("Predicted Frequency")
        plt.ylabel("True Frequency")
        plt.show()

    # Set trading threshold based on long confidence quantile
    return pred, corresponding_outcomes_signed, [prob_true_signed, prob_pred_signed]


def calculate_return(df, probabilities, directions, threshold=.5,  confidence_multiplier=1, side="both", use_select_hours=np.arange(25), display_plots=True):

    assert side in ["long", "short", "both"]

    equity_history = []
    time_history = []
    trade_indices = []
    trade_log_dict = {"Entry_Price (Close)": [], 'Entry_Time': [], 'Exit_Price (Close)': [], 'Exit_Time': [],
                      'Return': [], "Confidence": [], 'Trade_Direction': [], "Price_Direction": []}

    # Exclude the last trade since the last close for return calculations is missing (was dropped in feature building when return columns was shifted back for labeling).
    for i in range(len(directions) - 1):
        if side == "long":
            if directions[i] == 1 and probabilities[i][1] > threshold and df.index[i + 1].hour in use_select_hours:

                asset_return = df["Return"][i]

                confidence_boost = (probabilities[i][1] - threshold) * 100

                equity_history.append(1 + (asset_return * confidence_multiplier **
                                           confidence_boost))
                time_history.append(df.index[i])
                trade_indices.append(i)

                # Output trade log into csv for Evan
                trade_log_dict = trade_log(
                    trade_log_dict, asset_df=df, index=i, confidence=probabilities[i][1], trade_direction=1)

        elif side == "short":
            if directions[i] == 0 and probabilities[i][0] > threshold and df.index[i + 1].hour in use_select_hours:

                asset_return = df["Return"][i]

                confidence_boost = (probabilities[i][1] - threshold) * 100

                equity_history.append(1 - (asset_return * confidence_multiplier **
                                           confidence_boost))
                time_history.append(df.index[i])
                trade_indices.append(i)

                # Output trade log into csv for Evan
                trade_log_dict = trade_log(
                    trade_log_dict, asset_df=df, index=i, confidence=probabilities[i][1], trade_direction=-1)
        else:
            direction = directions[i]
            if probabilities[i][direction] < threshold:
                continue
            if direction == 0:
                asset_return = df["Return"][i] * -1
            else:
                asset_return = df["Return"][i]

            equity_history.append(1+asset_return)
            time_history.append(df.index[i])
            trade_indices.append(i)

    # Distinct Trades Calculation
    distinct_trades = len(distinct_trades_calculation(trade_indices))

    # Trades for evan

    trade_log_df = pd.DataFrame(trade_log_dict)
    trades_by_hour = tod_binning(trade_log_df)

    trade_log_df.to_csv('output_logs/trade_log.csv')

    # Create discting trade log, not by bar:
    dtl = distinct_trade_log(trade_log_df, side)
    dtl.to_csv('output_logs/distinct_trade_log.csv')

    backtest_accuracy = profitability_distibutions(
        trade_log_df, display_plots=display_plots, side=side)

    print("Number of Periods Traded: " + str(len(equity_history)))
    print("Disctinct Trades: " + str(distinct_trades))

    if display_plots:
        plt.plot(time_history, np.cumprod(equity_history), label="Model")
        plt.plot(df.index, np.cumprod(df["Return"] + 1), label="Benchmark")
        plt.title("Equity: " + side)
        plt.legend()
        plt.show()

    return trades_by_hour, backtest_accuracy


def distinct_trades_calculation(trade_indices):
    trade_indices = np.array(trade_indices)
    trade_index_gaps = np.diff(trade_indices)[1:]
    distinct_trade_indices = np.where(trade_index_gaps != 1)[0]
    return distinct_trade_indices


def distinct_trade_log(trade_log, side):

    ind = trade_log.index
    disctint_trade_log_dic = {"Entry": [trade_log["Exit_Time"][ind[0]]], "Exit": [
    ], "Entry_Price": [trade_log["Entry_Price (Close)"][ind[0]]], "Exit_Price": [], 'Return': [], "Trade_Direction": []}

    for n in range(len(ind) - 1):
        if trade_log["Exit_Time"][ind[n]] != trade_log["Entry_Time"][ind[n + 1]]:

            disctint_trade_log_dic["Exit"].append(
                trade_log["Exit_Time"][ind[n]])
            disctint_trade_log_dic["Exit_Price"].append(
                trade_log["Exit_Price (Close)"][ind[n]])

            if side == 'long':
                disctint_trade_log_dic['Return'].append(
                    disctint_trade_log_dic['Exit_Price'][-1] / disctint_trade_log_dic['Entry_Price'][-1])
                disctint_trade_log_dic['Trade_Direction'].append(1)
            else:
                disctint_trade_log_dic['Return'].append(
                    (2 * disctint_trade_log_dic['Entry_Price'][-1] - disctint_trade_log_dic['Exit_Price'][-1]) / disctint_trade_log_dic['Entry_Price'][-1])
                disctint_trade_log_dic['Trade_Direction'].append(-1)

            disctint_trade_log_dic["Entry"].append(
                trade_log["Entry_Time"][ind[n + 1]])
            disctint_trade_log_dic["Entry_Price"].append(
                trade_log["Entry_Price (Close)"][ind[n + 1]])

    disctint_trade_log_dic['Entry'].pop(-1)
    disctint_trade_log_dic['Entry_Price'].pop(-1)

    tl = pd.DataFrame(disctint_trade_log_dic)

    return tl


def trade_log(log_dict, asset_df, index, confidence, trade_direction):

    entry_price = asset_df["Close"][index]
    exit_price = asset_df["Close"][index+1]

    if trade_direction == 1:
        trade_return = exit_price / entry_price
    else:
        trade_return = (2 * entry_price - exit_price) / entry_price

    price_direction = np.sign(exit_price - entry_price)

    keys = list(log_dict.keys())
    vals = [entry_price, asset_df.index[index], exit_price,
            asset_df.index[index+1], trade_return, confidence, trade_direction, price_direction]
    for i in range(len(keys)):
        log_dict[keys[i]].append(vals[i])

    return log_dict


def profitability_distibutions(log_df, display_plots=True, side='long'):

    if side == 'long':
        profitable_indices = log_df["Price_Direction"] == 1
        unprofitable_indices = log_df["Price_Direction"] == -1
    else:
        profitable_indices = log_df["Price_Direction"] == -1
        unprofitable_indices = log_df["Price_Direction"] == 1

    p_success = sum(profitable_indices) / \
        (sum(profitable_indices) + sum(unprofitable_indices))
    print("Backtest Trade Accuracy: " + str(p_success))
    print("Average Profitable Trade: " +
          str(np.mean(log_df["Return"][profitable_indices] - 1)))
    print("Average Unprofitable Trade: " +
          str(np.mean(log_df["Return"][unprofitable_indices] - 1)))

    if display_plots:
        # Drop 0s in return column for log calculations
        log_df = log_df[log_df["Return"] != 0]

        # Create a figure and a set of subplots
        fig = plt.figure(figsize=(10, 6))

        # Create the first subplot in the first column (spanning 1 row and 1 column)
        ax1 = plt.subplot2grid((2, 2), (0, 0))

        # Create the second subplot in the first column, below the first subplot
        ax2 = plt.subplot2grid((2, 2), (1, 0))

        # Create the third subplot in the second column (spanning 2 rows)
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

        # Profitable PDF
        ax1.hist(np.log(log_df["Return"][profitable_indices]), bins=200)
        ax1.set_title('Profitable PDF')
        ax1.set_xlabel("Log Returns")
        ax1.set_ylabel("Counts")

        # Unprofitable PDF
        ax2.hist(-np.log(log_df["Return"][unprofitable_indices]),
                 bins=200)
        ax2.set_title('Unprofitable PDF')
        ax2.set_xlabel("Log Returns")
        ax2.set_ylabel("Counts")

        # Profitable CDF
        np_hist, np_bin_edges = np.histogram(
            np.log(log_df["Return"][profitable_indices]), bins='auto', density=True)

        # Reimann sum of weighted area under pdf * probability of profitable trades
        np_p_cdf = np.cumsum(p_success * np_hist *
                             np_bin_edges[:-1] * np.diff(np_bin_edges))

        ax3.plot(np_bin_edges[:-1], np_p_cdf, label="Profitable Predictions")

        # Unprofitable CDF. Same bin edges
        np_hist, _ = np.histogram(
            -np.log(log_df["Return"][unprofitable_indices]), bins=np_bin_edges, density=True)
        # Reimann sum of weighted area under pdf * probability of unprofitable trades
        np_up_cdf = np.cumsum((1 - p_success) * np_hist *
                              np_bin_edges[:-1] * np.diff(np_bin_edges))

        ax3.plot(np_bin_edges[:-1], np_up_cdf,
                 label="Unprofitable Predictions")
        ax3.plot(np_bin_edges[:-1], np_p_cdf - np_up_cdf,
                 label="Net Expectation of Model")
        ax3.set_ylabel("Log Return Expectation")
        ax3.set_xlabel("Log Return Magnitude")

        ax3.set_title("Expectation of Predictions")
        ax3.legend()
        plt.tight_layout()
        plt.show()

    return p_success


def tod_binning(trade_log):
    grouped = trade_log.groupby(trade_log["Exit_Time"].dt.hour)
    by_hr = {}
    for hour, group in grouped:
        by_hr[hour] = group[group["Price_Direction"] != 0]
    return by_hr


def analyze_tod(binned_trades, accuracies, returns, side):

    if side == 'long':
        profitable_direction = 1
        unprofitable_direction = -1
    else:
        profitable_direction = -1
        unprofitable_direction = 1

    chi_table = np.array([[(group["Price_Direction"] == profitable_direction).sum(), (group["Price_Direction"] == unprofitable_direction).sum()]
                          for hour, group in binned_trades.items()])

    try:
        chi2, p, dof, expected = chi2_contingency(chi_table)
        print("Probability that accuracies are equal across bins: " + str(p))
    except:
        pass

    accuracies_dict = {hour: [(group["Price_Direction"] == profitable_direction).sum() / len(group["Price_Direction"]), len(group["Price_Direction"])]
                       for (hour, group) in binned_trades.items()}
    accuracies_dict = dict(
        sorted(accuracies_dict.items(), key=lambda item: item[1]))

    print("Sorted By Accuracy: \n" + str(accuracies_dict))

    # Test difference in mean returns
    returns_by_hr = [group["Return"]
                     for hour, group in binned_trades.items()]
    try:
        f_statistic, p_value = f_oneway(*returns_by_hr)

        print("Probability that mean returns are equal across bins: " + str(p_value))
    except:
        pass

    returns_dict = {hour: group["Return"].mean()
                    for (hour, group) in binned_trades.items()}
    returns_dict = dict(
        sorted(returns_dict.items(), key=lambda item: item[1]))

    # print("Sorted By Return: \n" + str(returns_dict))

    # Overlay the calibrations
    if accuracies:
        slopes = {}
        for key, val in binned_trades.items():
            prob_true, prob_pred = calibration_curve(
                val["Price_Direction"], val["Confidence"], pos_label=profitable_direction, n_bins=5, strategy='quantile')
            # Regress and plot line
            slope, intercept, r_value, p_value, std_err = linregress(
                prob_pred, prob_true)
            slopes[key] = [prob_pred, slope * prob_pred +
                           intercept, statistics.mean(prob_true)]

        sorted_slopes = dict(
            sorted(slopes.items(), key=lambda item: item[1][2]))

        plt.figure(figsize=(10, 8))
        for i, (key, val) in enumerate(sorted_slopes.items(), 0):
            plt.subplot(2, 2, (i // 6) + 1)
            plt.plot(val[0], val[1], label="ET Hour: " + str(key))
            plt.legend()
            plt.title("Plot Number: " + str((i // 6) + 1))
            plt.xlabel("Predicted Frequency")
            plt.ylabel("True Frequency")

        plt.suptitle('Calibrations By Hour', fontsize=16)
        plt.tight_layout()
        plt.show()

    # Overlay Trade Expectations
    if returns:
        cdfs = {}
        for key, val in binned_trades.items():
            profitable_mag = np.log(val["Return"][val["Return"] > 1])
            unprofitable_mag = -np.log(val["Return"][val["Return"] < 1])
            p_success = len(profitable_mag) / \
                (len(profitable_mag) + len(unprofitable_mag))

            np_hist, np_bin_edges = np.histogram(
                unprofitable_mag, bins='auto', density=True)
            np_up_cdf = np.cumsum((1 - p_success) * np_hist *
                                  np_bin_edges[:-1] * np.diff(np_bin_edges))

            np_hist, _ = np.histogram(
                profitable_mag, bins=np_bin_edges, density=True)
            np_p_cdf = np.cumsum(p_success * np_hist *
                                 np_bin_edges[:-1] * np.diff(np_bin_edges))

            cdfs[key] = [np_bin_edges[:-1], np_p_cdf - np_up_cdf,
                         statistics.mean(np.log(val["Return"]))]

        sorted_cdfs = dict(
            sorted(cdfs.items(), key=lambda item: item[1][2]))

        plt.figure(figsize=(10, 8))
        for i, (key, val) in enumerate(sorted_cdfs.items(), 0):
            plt.subplot(2, 2, (i // 6) + 1)
            plt.plot(val[0], val[1], label="ET Hour: " + str(key))
            plt.legend()
            plt.title("Plot Number: " + str((i // 6) + 1))
            plt.xlabel("Log Return Magnitude")
            plt.ylabel("Log Return Expectation")

        plt.suptitle('Expectations By Hour', fontsize=16)
        plt.tight_layout()
        plt.show()
