from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster, dendrogram
from scipy.stats import pearsonr
from .feature_tools import custom_interp
from data_format_tools import load_price, make_multi_index
from .feature_building import price_features


def execute_PCA(features_df, feature_names, end, asset, max_depth, explained_var, run_spearman=False, plot_heat=False):
    # Correlation Heatmaps and Comparison
    pearson_cm = pearson_heat(
        data=features_df, features=feature_names, train_end=end, plot=plot_heat)
    if run_spearman:
        spearman_cm = spearman_heat(
            data=features_df, features=feature_names, train_end=end, plot=plot_heat)
        cm_diff(spearman_cm, pearson_cm)

    # Hierarchical Clustering
    clustered_features = cm_clustering(pearson_cm, max_depth=max_depth)

    PCA_selector = PCA_selection(explained_var=explained_var, scree_plot=False)

    pc_names = []
    for i, cluster in enumerate(clustered_features):
        features_df, pcs = PCA_selector.create_features(
            data=features_df, features=cluster, train_end=end, cluster_num=i, asset=asset)
        pc_names.extend(pcs)

    return features_df, pc_names, clustered_features


def LASSO_filter(df, features, train_end):

    df_test = df[df.index < train_end]

    logistic_model = LogisticRegressionCV(penalty="l1",
                                          solver="saga",
                                          max_iter=100, cv=10, random_state=0,
                                          n_jobs=-1)
    x_batch = df_test[features]

    correlation_matrix = x_batch.corr()
    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', mask=mask)
    plt.show()

    y_batch = np.sign(df_test["Return"].values)
    y_batch = np.where(y_batch == 0, 0, (y_batch + 1) / 2)

    logistic_model.fit(x_batch, y_batch)

    for i in range(len(features)):
        if logistic_model.coef_[0][i] == 0:
            print("Removed Feature: " + str(features[i]))
            features.remove(features[i])

    return features


class PCA_selection:

    def __init__(self, explained_var, scree_plot):
        self.ev = explained_var
        self.scree_plot = scree_plot

    def create_features(self, data, features, train_end, cluster_num, asset):

        all_data = data[features]
        all_data = all_data.loc[~(all_data == 0).all(axis=1)]

        train_data = all_data[all_data.index < train_end]

        scaler = StandardScaler()

        scaled_train_data = scaler.fit_transform(train_data)
        scaled_all_data = scaler.transform(all_data)

        # Applying PCA
        pca = PCA(n_components=self.ev)
        pca.fit(scaled_train_data)

        # Transforming the data
        pca_result = pca.transform(scaled_all_data)

        # Scree Plot
        explained_variance = pca.explained_variance_ratio_
        if self.scree_plot:

            plt.figure(figsize=(8, 4))
            plt.bar(range(1, len(explained_variance) + 1), explained_variance,
                    alpha=0.5, align='center', label='Individual explained variance')
            plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance),
                     where='mid', label='Cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal component index')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

        new_features = [asset + '_CN_' + str(cluster_num) +
                        "_PC_" + str(i) for i in range(len(explained_variance))]
        pc_features = pd.DataFrame(
            pca_result, index=all_data.index, columns=new_features)

        for feature in new_features:

            training_range_df = pc_features[feature][pc_features.index < train_end]

            pc_features[feature] = custom_interp(
                df_limits=training_range_df, df=pc_features[feature])

        # Restore the original index and fill with 0s
        pc_features = pc_features.reindex(data.index).fillna(0)
        pc_features = make_multi_index(pc_features)

        new_features_df = pd.concat([data, pc_features], axis=1)

        return new_features_df, new_features


def pearson_heat(data, features, train_end, plot=True):

    # For dfs filled with 0 to match index

    train_data = data[features][data.index < train_end]
    train_data = train_data.loc[~(train_data == 0).all(axis=1)]

    correlation_matrix = train_data.corr()

    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))

    if plot:
        sns.heatmap(correlation_matrix, annot=False,
                    cmap='coolwarm', mask=mask)
        plt.title("Pearson Heatmap")
        plt.show()

    return correlation_matrix


def spearman_heat(data, features, train_end, plot=True):

    # For dfs filled with 0 to match index
    data = data.loc[~(data == 0).all(axis=1)]

    train_data = data[features][data.index < train_end]

    correlation_matrix = train_data.corr(method='spearman')

    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))

    if plot:
        sns.heatmap(correlation_matrix, annot=False,
                    cmap='coolwarm', mask=mask)
        plt.title("Spearman Heatmap")
        plt.show()

    return correlation_matrix


def cm_diff(cm1, cm2):
    diff_m = cm1 - cm2
    mask = np.tril(np.ones_like(diff_m, dtype=bool))
    sns.heatmap(diff_m, annot=False, cmap='coolwarm', mask=mask)
    plt.title("Matrix Difference Heatmap")
    plt.show()


def cm_clustering(cm, max_depth):

    cm = 1 - cm

    Z = linkage(cm, method='average', metric="correlation")

    # Set a threshold to define the maximum depth, this might need some experimentation
    clusters = fcluster(Z, max_depth, criterion='distance')

    # Map the original data to these cluster labels
    cm['Cluster'] = clusters

    # Sort the data by clusters (optional, but can make the visualization clearer)
    sorted_cm = cm.sort_values('Cluster')

    # Get the names of clustered features
    clustered_features = []
    for i in range(min(clusters), max(clusters)+1):
        clustered_features.append(
            list(sorted_cm[sorted_cm['Cluster'] == i].index))

    # Remove the cluster column before plotting
    cluster_col = sorted_cm['Cluster']
    sorted_cm = sorted_cm.drop('Cluster', axis=1)
    sorted_cm = sorted_cm.transpose()

    # Visualize with Seaborn's clustermap
    hm = sns.heatmap(sorted_cm, annot=False, cmap='coolwarm')

    # Annotate with cluster numbers
    for idx, cluster_num in enumerate(cluster_col):
        hm.text(idx + 0.5, -0.5, cluster_num,
                ha='center', va='center', color='black', fontsize=5)

    plt.title("Sorted Feature Clusters", pad=15)
    plt.show()

    return clustered_features


# Assuming df is your DataFrame
def calculate_pvalues(df):
    df_cols = df.columns
    n = len(df_cols)
    p_values_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                p_values_matrix[i, j] = 0
            else:
                _, p_value = pearsonr(df[df_cols[i]], df[df_cols[j]])
                p_values_matrix[i, j] = p_value

    return pd.DataFrame(p_values_matrix, index=df_cols, columns=df_cols)
