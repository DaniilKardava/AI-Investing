from datetime import datetime
from data_format_tools import load_filtered_news, binary_ohe, multiclass_ohe, time_features,  global_exchange_status, alex_crypto_sentiment
from features.feature_selection import execute_PCA
from features.feature_building import combined_price_features
from model_building_tools import build_models, evaluate_models

# Training Dates. In practice it will start at the earliest available day.
start = datetime(2000, 1, 1)
end = datetime(2022, 1, 1)

feature_names = []
run_PCA = False
bin_features = False
feature_bins = 10
use_sentiment = False

main_directory = 'data/QQQ_1hour+.csv'
main_asset = 'QQQ'
assets = {main_asset: [main_directory]}  # , 'DXY': ["data/DXY_15min.csv"],
#          'TLT': ["data/TLT_15min.csv"], 'VIX': ["data/VIX_15min.csv"], "SPY": ["data/SPY_15min.csv"]}
use_indicators = ["RSI", "SMA", "MACD", "BBANDS", "DO"]
use_indicators = []
features_df, price_feature_names = combined_price_features(
    assets=assets, main_asset=main_asset, end=end, bin_features=bin_features, feature_bins=feature_bins, run_PCA=run_PCA, use_indicators=use_indicators)

if use_sentiment:
    sentiment_path = 'data/crypto_sentiment.csv'
    features_df, sentiment_feature_names = alex_crypto_sentiment(
        sentiment_path, interval="1H", features_df=features_df)
    feature_names.extend(sentiment_feature_names)

markets = ["NYSE", "LSE", "JPX", "HKEX", "ASX", "TSX", "SSE"]

features_df = binary_ohe(data=features_df, threshold_percentile=0)
features_df, time_feature_names = time_features(features_df, encoding='onehot')
features_df = global_exchange_status(features_df, markets)
# features_df = multiclass_ohe(
#     data=features_df, bins=[-1, -.5, -.25, -.1, 0, .1, .25, .5, 1])

feature_names.extend(price_feature_names)
# feature_names.extend(markets)
# feature_names.extend(time_feature_names)

clustered_features = []
if run_PCA:
    # PCA Assets Individually
    for asset, vals in assets.items():
        # Gather price features for this asset
        sub_feature_names = [
            feat for feat in price_feature_names if feat.split("_")[0] == asset]
        # Remove them from the main features list
        feature_names = [
            feat for feat in feature_names if feat not in sub_feature_names]

        features_df, sub_feature_names, sub_clustered_features = execute_PCA(
            features_df=features_df, feature_names=sub_feature_names, end=end, asset=asset, max_depth=.1, explained_var=.9)

        feature_names.extend(sub_feature_names)
        clustered_features.extend(sub_clustered_features)


print(features_df)
print(feature_names)

# I believe the format is 9: 9.00-9.59 trades
trade_select_hours = [9, 10, 11, 12, 13, 14, 15]
summary_name = 'intraday_OOS_summary'
# summary_name = 'OOS_summary'

# Creating models and evaluating models
path = "saved_models/QQQ/w_1_8_custom_bin_price_QQQ_1hour_2022_1/"
# build_models(path=path, features_df=features_df, feature_names=feature_names, start=start,
#              end=end, run_PCA=run_PCA, clustered_features=clustered_features, num_models=15, import_last=False, long_only=False, window_size=1)
evaluate_models(path=path, features_df=features_df,
                feature_names=feature_names, start=datetime(2022, 1, 1), end=datetime(2023, 12, 1), side='long', confidence_percentile=75, display_plots=False, trade_select_hours=trade_select_hours, start_with=0, window_size=1, summary_name=summary_name)
