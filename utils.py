import pandas as pd
import numpy as np


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X)


def check_ifreal(y: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(y)


def entropy(Y: pd.Series) -> float:
    values, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))


def gini_index(Y: pd.Series) -> float:
    values, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


def mse(Y: pd.Series) -> float:
    mean_val = Y.mean()
    return np.mean((Y - mean_val) ** 2)


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    if criterion == "information_gain":
        base = entropy(Y)
    elif criterion == "gini_index":
        base = gini_index(Y)
    elif criterion == "mse":
        base = mse(Y)
    else:
        raise ValueError("Unknown criterion")

    values, counts = np.unique(attr, return_counts=True)
    weighted = 0
    for v, c in zip(values, counts):
        y_sub = Y[attr == v]
        if criterion == "information_gain":
            weighted += (c / len(attr)) * entropy(y_sub)
        elif criterion == "gini_index":
            weighted += (c / len(attr)) * gini_index(y_sub)
        else:
            weighted += (c / len(attr)) * mse(y_sub)
    return base - weighted if criterion != "mse" else base - weighted


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    best_gain = -1e9
    best_attr = None
    for feature in features:
        gain = information_gain(y, X[feature], criterion)
        if gain > best_gain:
            best_gain = gain
            best_attr = feature
    return best_attr


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    mask = X[attribute] == value
    return X[mask].drop(columns=[attribute]), y[mask], X[~mask].drop(columns=[attribute]), y[~mask]
