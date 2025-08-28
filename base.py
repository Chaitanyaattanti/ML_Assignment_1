import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold

np.random.seed(42)

# Dataset generation

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=2, class_sep=0.5
)

# Convert to pandas for our custom tree
X = pd.DataFrame(X, columns=["x1", "x2"])
y = pd.Series(y, dtype="category")

# Plot dataset
plt.scatter(X["x1"], X["x2"], c=y)
plt.title("Generated dataset")
plt.show()


# Q2 (a) Train/test split 70/30

def train_and_evaluate(X, y, max_depth=5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    tree = DecisionTree(criterion="information_gain", max_depth=max_depth)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    print("=== Q2 (a) Train/Test Split ===")
    print("Accuracy:", accuracy(y_pred, y_test))
    for cls in y_test.unique():
        print(f"Class {cls}: Precision={precision(y_pred, y_test, cls)}, Recall={recall(y_pred, y_test, cls)}")


# Q2 (b) Nested cross-validation

def nested_cross_validation(X, y, depths=range(1, 11), outer_splits=5):
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    best_depths = []
    outer_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
        avg_scores = {}

        for d in depths:
            scores = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
                X_inner_train, X_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
                y_inner_train, y_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]

                tree = DecisionTree(criterion="information_gain", max_depth=d)
                tree.fit(X_inner_train, y_inner_train)
                preds = tree.predict(X_val)
                scores.append(accuracy(preds, y_val))

            avg_scores[d] = np.mean(scores)

        # pick best depth for this fold
        best_d = max(avg_scores, key=avg_scores.get)
        best_depths.append(best_d)

        # retrain with best_d
        best_tree = DecisionTree(criterion="information_gain", max_depth=best_d)
        best_tree.fit(X_train, y_train)
        final_preds = best_tree.predict(X_test)
        final_acc = accuracy(final_preds, y_test)
        outer_accuracies.append(final_acc)

        print(f"Fold {fold}: Best depth = {best_d}, Accuracy = {final_acc}")

    opt_depth = max(set(best_depths), key=best_depths.count)
    print("\n=== Q2 (b) Nested CV Results ===")
    print("Per-fold accuracies:", outer_accuracies)
    print("Average accuracy:", np.mean(outer_accuracies))
    print("Optimum depth (most frequent):", opt_depth)

# Run experiments

train_and_evaluate(X, y)
nested_cross_validation(X, y)
