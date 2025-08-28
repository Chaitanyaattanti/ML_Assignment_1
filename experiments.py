import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from base import DecisionTree

np.random.seed(42)
num_average_time = 10  # number of runs for averaging

# Generate fake dataset

def generate_data(N, M, input_type="real", output_type="real"):
    if input_type == "real":
        X = pd.DataFrame(np.random.randn(N, M))
    else:  # discrete input
        X = pd.DataFrame(np.random.randint(0, 2, size=(N, M))).astype("category")

    if output_type == "real":
        y = pd.Series(np.random.randn(N))
    else:  # discrete output
        y = pd.Series(np.random.randint(0, 2, size=N), dtype="category")

    return X, y

# Measure training and prediction times

def measure_time(N_values, M_values, tree_cases):
    results = {case: {"train": [], "predict": []} for case in tree_cases}

    for N in N_values:
        for M in M_values:
            for case in tree_cases:
                input_type, output_type = case.split("->")
                X, y = generate_data(N, M, input_type, output_type)

                # Train-test split: 70%-30%
                X_train, y_train = X.iloc[:int(0.7*N)], y.iloc[:int(0.7*N)]
                X_test, y_test = X.iloc[int(0.7*N):], y.iloc[int(0.7*N):]

                train_times, predict_times = [], []

                for _ in range(num_average_time):
                    clf = DecisionTree(criterion="information_gain", max_depth=10)
                    start = time.time()
                    clf.fit(X_train, y_train)
                    train_times.append(time.time() - start)

                    start = time.time()
                    clf.predict(X_test)
                    predict_times.append(time.time() - start)

                avg_train = np.mean(train_times)
                avg_predict = np.mean(predict_times)

                results[case]["train"].append(avg_train)
                results[case]["predict"].append(avg_predict)

                print(f"N={N}, M={M}, {case} | Train={avg_train:.5f}s, Predict={avg_predict:.5f}s")

    return results

# Plot results

def plot_results(N_values, M_values, results, mode="train"):
    plt.figure(figsize=(12,6))
    for case in results:
        plt.plot(range(len(results[case][mode])), results[case][mode], marker="o", label=case)

    xticks = [f"N={N},M={M}" for N in N_values for M in M_values]
    plt.xticks(range(len(xticks)), xticks, rotation=45)
    plt.ylabel("Time (seconds)")
    plt.title(f"Decision Tree {mode.capitalize()} Time vs Dataset Size")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main experiment

if __name__ == "__main__":
    N_values = [200, 500]  # vary samples
    M_values = [5, 10]       # vary features

    tree_cases = [
        "discrete->discrete",
        "real->discrete",
        "discrete->real",
        "real->real"
    ]

    results = measure_time(N_values, M_values, tree_cases)

    print("\n=== Training Time Plot ===")
    plot_results(N_values, M_values, results, mode="train")

    print("\n=== Prediction Time Plot ===")
    plot_results(N_values, M_values, results, mode="predict")

    print("\nTheoretical Complexity:")
    print("Training ≈ O(N × M × log N)")
    print("Prediction ≈ O(depth × N_test), depth ≈ O(log N)")
