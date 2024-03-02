import pandas as pd
from sklearn.calibration import calibration_curve
import numpy as np
from matplotlib import pyplot as plt
from tools.feature_tools import rolling_accuracy
import yfinance as yf

# Load data
model_predictions = pd.read_csv('logs/Rolling.csv', index_col=0)

model_predictions.index = pd.to_datetime(model_predictions.index)

CALIBRATION_BINS = 7

# Get baseline success rate
print(
    f'Baseline Success Rate: {model_predictions["True Direction"].sum() / len(model_predictions["True Direction"])}')

# Convert predictions to binary
model_predictions['Direction Prediction'] = (
    model_predictions['Model Prediction'] > .5).astype(int)

# Graph rolling accuracy
rolling_accuracy(model_predictions['True Direction'],
                 model_predictions['Direction Prediction'], period=10)

# Analyze shorts
short_trades = model_predictions.loc[model_predictions['Direction Prediction'] == 0]

short_success = short_trades['Direction Prediction'] == short_trades['True Direction']

print(
    f'Short success rate for {short_success.size} trades: {short_success.sum() / short_success.size}')


# Analyze calibration confidences
prob_true, prob_pred = calibration_curve(
    short_trades['True Direction'], 1 - short_trades['Model Prediction'], pos_label=0, n_bins=CALIBRATION_BINS, strategy='quantile')

plt.scatter(prob_pred, prob_true)
plt.title('Short Calibration')
plt.xlabel("Predicted Frequency")
plt.ylabel("True Frequency")
plt.show()


# Analyze long
long_trades = model_predictions.loc[model_predictions['Direction Prediction'] == 1]

long_success = long_trades['Direction Prediction'] == long_trades['True Direction']

print(
    f'Long success rate for {long_success.size} trades: {long_success.sum() / long_success.size}')


# Analyze calibration confidences
prob_true, prob_pred = calibration_curve(
    long_trades['True Direction'], long_trades['Model Prediction'], pos_label=1, n_bins=CALIBRATION_BINS, strategy='quantile')

plt.scatter(prob_pred, prob_true)
plt.title('Long Calibration')
plt.xlabel("Predicted Frequency")
plt.ylabel("True Frequency")
plt.show()

'''
Graph a heatmap of long confidences.
'''
ROLLING = 1

spy_history = yf.download('SPY', long_trades.index.min, long_trades.index.max)

plt.plot(spy_history.index, spy_history.close)
plt.scatter(long_trades.index, long_trades['Model Prediction'])
plt.show()


'''
Consider the performance of the most confident predictions
'''

THRESHOLD = .54
confident_trades = long_trades.loc[long_trades['Model Prediction'] > THRESHOLD]
print(confident_trades)
prediction_outcome = confident_trades['Direction Prediction'] == confident_trades['True Direction']

print(
    f'Number of trades with confidence greater than {THRESHOLD}: {confident_trades.index.size}')
print(
    f'Accuracy of confident long trades: {prediction_outcome.sum() / prediction_outcome.index.size}')
