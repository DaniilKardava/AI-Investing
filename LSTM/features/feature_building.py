
import numpy as np
from finta import TA
import pandas as pd
import warnings
from indicator_categories import pct_change_normalization, price_normalization, root_price_normalization, standardized
from .feature_tools import lookahead_indicator, accomodate_warmup, compare_lengths, custom_interp, one_hot_inputs, median_binning, volume_indicator, mag_returns, sign_return, bin_return
from datetime import datetime
from data_format_tools import load_price
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


def price_features(asset, asset_df, training_end, bin_features=False, feature_bins=30, PCA_enabled=False, use_indicators=None):

    # Name change
    features_df = asset_df

    # Rename the previous return feature to make it distinguishable in the final df
    features_df.rename(
        columns={"Return_Unshifted": asset + "_Return_Unshifted"}, inplace=True)

    # Add basic return as the fundamental feature
    indicator_names = []

    # Try to make return feature signed
    q_eight_bins = [0, 0.05, .2, .35, .5, .65, .8, .95, 1]
    q_ten_bins = [0, 0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    c_eight_bins = [-np.inf, -.75, -.25, -.1, 0, .1, .25, .75, np.inf]

    features_df, indicator_names = bin_return(
        features_df, training_end, asset + "_Return_Unshifted", c_eight_bins, indicator_names, bin_type='custom')
    '''features_df, indicator_names = sign_return(
        features_df, asset + "_Return_Unshifted", indicator_names)'''
    '''if np.sum(features_df['Volume']) != 0:
        features_df, indicator_names = volume_indicator(
            features_df, asset + '_Standard_Volume', indicator_names)'''

    # Create a prices df with a single index that is supported by FinTa package
    single_level_asset_df = asset_df.copy(deep=True)
    single_level_asset_df.columns = single_level_asset_df.columns.droplevel(1)

    # Add all indicators to the df:
    indicators = standardized + pct_change_normalization + \
        price_normalization + root_price_normalization

    if use_indicators == None:
        use_indicators = indicators

    # Set maximum accomodation date for indicator warm ups.
    warmup_cut = asset_df.index[100]

    for indicator in use_indicators:

        method = getattr(TA, indicator)
        try:
            indicator_df = method(single_level_asset_df)
        except:
            continue

        # Convert to standard df form:
        if isinstance(indicator_df, pd.Series):
            indicator_df = indicator_df.to_frame()

        # Some indicators are boolean
        indicator_df = indicator_df.astype(float)

        for indicator_component in list(indicator_df):

            # Give it a unique reference to prevent overwriting columns with same sub names
            indicator_component_reference = indicator + \
                '_' + str(indicator_component)
            indicator_component_df = indicator_df[indicator_component]

            # Drop any indicator with lookahead:
            if lookahead_indicator(indicator_component_df, indicator_component_reference):
                continue

            # Test whether to accomodate warmup period for lagging indicator:
            fvi = accomodate_warmup(indicator_component_df)
            if fvi is not None and fvi < warmup_cut:
                features_df = features_df[features_df.index >= fvi]
            else:
                print("Warmup Period too Long: " +
                      str(indicator_component_reference) + ", ends on: " + str(fvi))
                continue

            # Dropna
            indicator_component_df.dropna(inplace=True)

            # Replace old stationarity test with appropriate standardization
            if indicator in price_normalization:
                aligned_close = asset_df["Close"].reindex(
                    indicator_component_df.index)
                indicator_component_df = indicator_component_df / aligned_close
            elif indicator in root_price_normalization:
                aligned_close = asset_df["Close"].reindex(
                    indicator_component_df.index)
                indicator_component_df = indicator_component_df / \
                    np.sqrt(aligned_close)
            elif indicator in pct_change_normalization:
                indicator_component_df = indicator_component_df.pct_change()
                # Inf shows up on division by 0, often in volume indicators with volume data missing
                if indicator_component_df.max().max() == np.inf:
                    print("Inf found in: " + str(indicator_component_reference))
                    continue

                fvi = accomodate_warmup(indicator_component_df)
                if fvi is not None and fvi < warmup_cut:
                    features_df = features_df[features_df.index >= fvi]
                else:
                    print("Warmup Period too Long: " +
                          str(indicator_component_reference) + ", ends on: " + str(fvi))
                    continue

            indicator_component_df = indicator_component_df[(indicator_component_df.index >= features_df.index[0]) &
                                                            (indicator_component_df.index <= features_df.index[-1])]

            if not compare_lengths(indicator_component_df, features_df):
                print("Lengths don't match for indicator: " +
                      str(indicator_component_reference))
                continue

            # Min max inputs if not using PCA, otherwise let pca class take care of scaling.
            if PCA_enabled:
                indicator_component_np = indicator_component_df.to_numpy()

                features_df[asset + '_' +
                            indicator_component_reference] = indicator_component_np
            else:

                training_range_df = indicator_component_df[indicator_component_df.index < training_end]

                if bin_features:

                    # Median Binning
                    indicator_component_np = median_binning(
                        training_range_df, indicator_component_df, feature_bins)
                    features_df[asset + '_' +
                                indicator_component_reference] = indicator_component_np

                    # One Hot Binning
                    '''multi_index_name = asset + '_' + indicator_component_reference
                    multi_index, ohe_data = one_hot_inputs(
                        training_range_df=training_range_df, indicator_component_df=indicator_component_df, multi_index_name=multi_index_name, feature_bins=feature_bins)
                    ohe_feature_df = pd.DataFrame(
                        ohe_data, index=indicator_component_df.index, columns=multi_index)
                    
                    features_df = pd.concat(
                        [features_df, ohe_feature_df], axis=1)'''

                else:
                    indicator_component_np = custom_interp(
                        df_limits=training_range_df, df=indicator_component_df)

                    features_df[asset + '_' +
                                indicator_component_reference] = indicator_component_np

            indicator_names.append(asset + '_' +
                                   indicator_component_reference)

    return features_df, indicator_names


def combined_price_features(assets, main_asset, end, bin_features, feature_bins, run_PCA, use_indicators):

    # Load all datasets
    truncate_date = []
    for asset, items in assets.items():

        asset_path = items[0]
        asset_df = load_price(path=asset_path)

        assets[asset].append(asset_df)
        truncate_date.append(asset_df.index[0])

    truncate_date = max(truncate_date)

    feature_names = []
    for asset, items in assets.items():
        print("ASSET: " + str(asset))
        asset_df = items[1]
        asset_df = asset_df[asset_df.index > truncate_date]
        features_df, asset_feature_names = price_features(asset=asset,
                                                          asset_df=asset_df, training_end=end, bin_features=bin_features, feature_bins=feature_bins, PCA_enabled=run_PCA, use_indicators=use_indicators)

        if asset == main_asset:
            assets[asset][1] = features_df
        else:
            assets[asset][1] = features_df[asset_feature_names]

        feature_names.extend(
            asset_feature_names)

    base_df = assets[main_asset][1]
    feature_dfs = [val[1].reindex(base_df.index)
                   for key, val in assets.items()]

    features_df = pd.concat(feature_dfs, axis=1)

    features_df.fillna(0, inplace=True)

    return features_df, feature_names
