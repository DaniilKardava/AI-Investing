import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix


def conditional_entropy(X, y):
    """
    Calculate the conditional entropy of each feature in X given the classes in y.
    Parameters:
    X (pd.DataFrame): DataFrame containing the features.
    y (pd.Series): Series containing the class labels.
    Returns:
    pd.DataFrame: A DataFrame with conditional entropies of each feature for each class.
    """
    classes = np.unique(y)
    conditional_entropies = pd.DataFrame(index=X.columns, columns=classes)
    for feature in X.columns:
        for cls in classes:
            feature_subset = X[feature][y == cls]
            feature_entropy = entropy(
                feature_subset.value_counts(normalize=True), base=2)
            conditional_entropies.loc[feature, cls] = feature_entropy
    return conditional_entropies


def transform_features_with_permutation_entropy(X, m, tau):
    """
    Transform features using permutation entropy.
    Parameters:
    X (np.ndarray): The input array containing features.
    m (int): The embedding dimension.
    tau (int): The delay time.
    Returns:
    np.ndarray: Transformed feature array.
    """
    n, num_features = X.shape
    transformed_features = np.zeros_like(X)
    for i in range(num_features):
        for j in range(n - m * tau + 1):
            permutation_vector = X[j:j + m * tau:tau, i]
            transformed_features[j, i] = np.argsort(
                permutation_vector).dot(2 ** np.arange(m)[::-1])
    return transformed_features


def compute_permutation_entropy(X, m_range, tau):
    """
    Compute permutation entropy for a range of embedding dimensions.
    Parameters:
    X (np.ndarray): The input array containing features.
    m_range (range): The range of embedding dimensions.
    tau (int): The delay time.
    Returns:
    list: List of entropies for each dimension in m_range.
    """
    entropies = []
    for m in m_range:
        entropies.append(
            np.mean([entropy(np.histogram(X[:, i], bins=m)[0]) for i in range(X.shape[1])]))
    return entropies


def compute_first_second_derivatives(values):
    """
    Compute the first and second discrete derivatives of an array.
    Parameters:
    values (np.ndarray): The input array.
    Returns:
    tuple: A tuple containing arrays of first and second derivatives.
    """
    first_derivative = np.diff(values, n=1)
    second_derivative = np.diff(values, n=2)
    return first_derivative, second_derivative


def preprocess_data(file_path):
    """
    Load and preprocess data.
    Parameters:
    file_path (str): Path to the data file.
    Returns:
    tuple: Tuple containing preprocessed feature arrays and labels.
    """
    data = pd.read_csv(file_path)
    target = 'your_target'  # Your target column
    X = data.drop(target, axis=1)
    y = data[target]
    scaler = StandardScaler()
    return scaler, X, y


def adjust_and_scale_data(X, y, scaler):
    """
    Adjust for class imbalance and scale data.
    Parameters:
    X (pd.DataFrame): DataFrame containing features.
    y (pd.Series): Series containing labels.
    scaler (StandardScaler): An instance of StandardScaler for scaling data.
    Returns:
    np.ndarray: Adjusted and scaled feature array.
    """
    X_conditional_entropy = conditional_entropy(X, y)
    class_proportions = y.value_counts(normalize=True)
    adjusted_entropy = X_conditional_entropy.copy()
    for cls in class_proportions.index:
        adjusted_entropy[cls] *= (1 / class_proportions[cls])
    X_adjusted = X.copy()
    for cls in class_proportions.index:
        class_indices = y == cls
        X_adjusted.loc[class_indices] = X.loc[class_indices] * \
            adjusted_entropy[cls]
    return scaler.fit_transform(X_adjusted)


def select_features(X, y, n_features, mi_weight=0.5, entropy_weight=0.5):
    """
    Select top features based on a weighted combination of mutual information and entropy,
    specifically designed for adjusted and scaled data.
    Parameters:
    X (np.ndarray): Pre-processed feature array (adjusted and scaled).
    y (pd.Series): Series containing labels.
    n_features (int): Number of top features to select.
    mi_weight (float): Weight for mutual information in the feature ranking.
    entropy_weight (float): Weight for entropy in the feature ranking.
    Returns:
    np.ndarray: Array of selected features.
    Raises:
    ValueError: If n_features exceeds the number of features in X.
    """
    # Error handling
    if n_features > X.shape[1]:
        raise ValueError(
            "Number of features to select cannot exceed the total number of features")
    mutual_info = mutual_info_classif(X, y)
    feature_entropies = np.apply_along_axis(entropy, 0, X)
    # Normalize the mutual information and entropy values
    scaler = MinMaxScaler()
    mutual_info_normalized = scaler.fit_transform(
        mutual_info.reshape(-1, 1)).flatten()
    entropy_normalized = scaler.fit_transform(
        feature_entropies.reshape(-1, 1)).flatten()
    # Combine the metrics with weights
    combined_score = mi_weight * mutual_info_normalized + \
        entropy_weight * (1 - entropy_normalized)
    # Rank features based on the combined score
    ranked_features = np.argsort(combined_score)[::-1]
    # Select top features
    selected_features = ranked_features[:n_features]
    return X[:, selected_features]


def main():
    # Load and preprocess the data
    file_path = ""  # Load your data
    scaler, X, y = preprocess_data(file_path)
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    # Adjust for class imbalance and scale
    X_train_adjusted_scaled = adjust_and_scale_data(X_train, y_train, scaler)
    X_test_adjusted_scaled = adjust_and_scale_data(X_test, y_test, scaler)
    # Transform features using permutation entropy
    m_range = range(1, 10)
    optimal_delay = 6  # Adjust this value based on your data
    permutation_entropies = compute_permutation_entropy(
        X_train_adjusted_scaled, m_range, optimal_delay)
    first_derivative, second_derivative = compute_first_second_derivatives(
        permutation_entropies)
    optimal_dimension = next((i for i, second_deriv in enumerate(
        second_derivative, start=2) if second_deriv < 0), m_range[-1])
    X_train_transformed = transform_features_with_permutation_entropy(
        X_train_adjusted_scaled, optimal_dimension, optimal_delay)
    X_test_transformed = transform_features_with_permutation_entropy(
        X_test_adjusted_scaled, optimal_dimension, optimal_delay)
    # Feature Selection
    n_features = 10  # Adjust this value based on your data
    X_train_transformed_selected = select_features(
        X_train_transformed, y_train, n_features)
    X_test_transformed_selected = select_features(
        X_test_transformed, y_test, n_features)
