from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from utils import *


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, output=None):
        self.feature = feature
        self.threshold = threshold   # threshold for regression splits
        self.left = left
        self.right = right
        self.output = output


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # classification only
    max_depth: int = 5

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_regression = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.is_regression = check_ifreal(y)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or depth >= self.max_depth or X.empty:
            return Node(output=y.mean() if self.is_regression else y.mode()[0])

        features = X.columns
        crit = "mse" if self.is_regression else self.criterion
        best_attr = opt_split_attribute(X, y, crit, features)

        if best_attr is None:
            return Node(output=y.mean() if self.is_regression else y.mode()[0])

        if self.is_regression:
            # choose threshold = median for continuous regression splits
            best_value = X[best_attr].median()
            X_left, y_left = X[X[best_attr] <= best_value], y[X[best_attr] <= best_value]
            X_right, y_right = X[X[best_attr] > best_value], y[X[best_attr] > best_value]
        else:
            # classification split: mode-based equality
            best_value = X[best_attr].mode()[0]
            X_left, y_left, X_right, y_right = split_data(X, y, best_attr, best_value)

        left_child = self._build_tree(X_left, y_left, depth + 1) if not y_left.empty else None
        right_child = self._build_tree(X_right, y_right, depth + 1) if not y_right.empty else None

        return Node(feature=best_attr, threshold=best_value, left=left_child, right=right_child)

    def _predict_row(self, row, node: Node):
        if node.output is not None:
            return node.output

        if self.is_regression:
            if row[node.feature] <= node.threshold:
                return self._predict_row(row, node.left) if node.left else node.output
            else:
                return self._predict_row(row, node.right) if node.right else node.output
        else:
            if row[node.feature] == node.threshold:
                return self._predict_row(row, node.left) if node.left else node.output
            else:
                return self._predict_row(row, node.right) if node.right else node.output

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(lambda row: self._predict_row(row, self.root), axis=1)

    def _print_tree(self, node, depth=0):
        if node is None:
            return
        indent = "    " * depth
        if node.output is not None:
            print(f"{indent}Output: {node.output}")
        else:
            if self.is_regression:
                print(f"{indent}?({node.feature} <= {node.threshold:.3f})")
            else:
                print(f"{indent}?({node.feature} == {node.threshold})")
            print(f"{indent}Y:")
            self._print_tree(node.left, depth + 1)
            print(f"{indent}N:")
            self._print_tree(node.right, depth + 1)

    def plot(self):
        self._print_tree(self.root)
