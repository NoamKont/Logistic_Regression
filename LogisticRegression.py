import numpy as np


class LogisticRegression:
    def __init__(self):
        self.weights_ = []  # weights[0] represent the 'b' and [1:] represent the weights vector.
        self.iterations = 10000
        self.learning_rate = 0.001
        self.thresh = 0.01
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, w, X, y):
        N, n = X.shape  # N: number of samples, n: number of features
        gradient = np.zeros(n)
        for i in range(N):
            x_i = X[i]
            y_i = y[i]
            gradient += (-y_i * x_i) / (self.sigmoid(y_i * np.dot(w, x_i)))

        gradient = gradient/N
        return gradient


    def fit(self, X, y):
        ones_column = np.ones((X.shape[0], 1), dtype=X.dtype)  # adding a '1' column to represent the 'b' in the module.
        X = np.hstack((ones_column, X))
        w = np.random.randn(X.shape[1])  # starting from a random vector
        # Gradient decent
        for i in range(self.iterations):
            g = self.gradient(w,X,y)  # g stand for gradient
            if(g > self.thresh):
                w = w - self.learning_rate*g

        self.weights_ = w

    def predict(self, X, add_bias=True):
        res = []
        if add_bias:  # if we already add bias in the other function we don't need to add another '1' column
            ones_column = np.ones((X.shape[0], 1), dtype=X.dtype)
            X = np.hstack((ones_column, X))
        for i in range(X.shape[0]):
            if(self.predict_proba(X[i]) > 0.5):
                res.append(1)
            else:
                res.append(0)

    def predict_proba(self, x):
        return self.sigmoid(np.dot(x, self.weights_))

    def compare_lists(self, list1, list2):
        return [1 if x == y else 0 for x, y in zip(list1, list2)]

    def score(self, X, y, add_bias=True):
        pred = self.predict(X, add_bias)
        compared = self.compare_lists(y, pred)
        return np.sum(compared) / len(y)


    @staticmethod
    def train_test_split(X, y, train_size=0.8):
        train_part = int(X.shape[0] * train_size)

        X_train = X[:train_part].values
        X_test = X[train_part:].values
        y_train = y[:train_part].values
        y_test = y[train_part:].values
        return X_train, X_test, y_train, y_test