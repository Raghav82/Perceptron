
import pandas as pd
import numpy as np

# ----------------------------
# Load & prepare Sonar dataset
# ----------------------------
def load_sonar():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
    df = pd.read_csv(url, header=None)

    # Last column is label: strings 'R' or 'M'
    # Explicit binary mapping: 'R' -> 0 (Rock), 'M' -> 1 (Mine)
    df[df.columns[-1]] = df[df.columns[-1]].map({'R': 0, 'M': 1})

    data = df.values  # NumPy array
    return data

# ----------------------------
# Perceptron implementation
# ----------------------------
def predict(row, bias_weights):
    """
    row: 1D array of length F+1 (features + label). We only use features here.
    bias_weights: 1D array of length F+1 where bias_weights[0] is bias,
                  and bias_weights[1:] are weights for F features.
    """
    bias = bias_weights[0]
    weights = bias_weights[1:]
    inputs = row[:-1]  # features only
    output = np.dot(weights, inputs) + bias
    return 1 if output >= 0 else 0

def train_weights(train, learning_rate=0.1, epochs=50, shuffle=True):
    """
    train: 2D array-like; each row: features + label (last element).
    learning_rate: eta for perceptron updates.
    epochs: number of passes through the dataset.
    shuffle: whether to shuffle rows each epoch.
    """
    n_cols = len(train[0])             # F + 1 (label)
    bias_weights = np.zeros(n_cols)    # [bias, w_1..w_F]
    for epoch in range(epochs):
        if shuffle:
            np.random.shuffle(train)
        total_error = 0
        for row in train:
            y_hat = predict(row, bias_weights)
            y = int(row[-1])
            error = y - y_hat
            total_error += error ** 2
            inputs = row[:-1]  # NumPy array of features
            # Update rule
            bias_weights[0] += learning_rate * error
            bias_weights[1:] += inputs * (learning_rate * error)
        # Optional: monitor error
        # print(f"Epoch {epoch+1}/{epochs} - SSE: {total_error}")
    return bias_weights

# ----------------------------
# Simple train/test evaluation
# ----------------------------
if __name__ == "__main__":
    data = load_sonar()
    # Split: 70% train, 30% test
    np.random.seed(42)
    idx = np.random.permutation(len(data))
    split = int(0.7 * len(data))
    train = data[idx[:split]]
    test = data[idx[split:]]

    w = train_weights(train, learning_rate=0.1, epochs=50, shuffle=True)

    # Evaluate
    def accuracy(dataset, w):
        correct = sum(1 for row in dataset if predict(row, w) == int(row[-1]))
        return correct / len(dataset)

    acc_train = accuracy(train, w)
    acc_test = accuracy(test, w)
    print(f"Train accuracy: {acc_train:.3f}")
    print(f"Test  accuracy: {acc_test:.3f}")
