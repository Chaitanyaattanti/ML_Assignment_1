import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree   
from metrics import *                
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

# Load dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
data = pd.read_csv(
    url,
    delim_whitespace=True,
    header=None,
    names=["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model year", "origin", "car name"]
)


# Remove "car name" (string column not useful for regression)
data = data.drop(columns=["car name"])

# Replace '?' in horsepower with NaN, convert to numeric
data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")

# Drop rows with missing values
data = data.dropna().reset_index(drop=True)

# Features and target
X = data.drop(columns=["mpg"])
y = data["mpg"]

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Q3 (a) Custom Decision Tree

print("=== Q3 (a) Custom Decision Tree ===")

# Your DecisionTree (assuming it supports regression via variance gain)
tree = DecisionTree(criterion="variance_gain", max_depth=5)
tree.fit(X_train, y_train)

y_pred_custom = tree.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred_custom))
print("MAE:", mean_absolute_error(y_test, y_pred_custom))
print("R^2:", r2_score(y_test, y_pred_custom))


# Q3 (b) Sklearn Decision Tree

print("\n=== Q3 (b) Sklearn Decision Tree ===")

sk_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
sk_tree.fit(X_train, y_train)

y_pred_sklearn = sk_tree.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred_sklearn))
print("MAE:", mean_absolute_error(y_test, y_pred_sklearn))
print("R^2:", r2_score(y_test, y_pred_sklearn))


# Compare visually

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="True MPG", marker="o")
plt.plot(y_pred_custom, label="Custom Tree", marker="x")
plt.plot(y_pred_sklearn, label="Sklearn Tree", marker="s")
plt.legend()
plt.title("Comparison of True vs Predicted MPG")
plt.show()
