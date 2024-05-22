import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class LogisticRegression:
    def __init__(self):
        self.weights_ = []  # weights[0] represent the 'b' and [1:] represent the weights vector.
        self.iterations = 5000
        self.learning_rate = 0.1
        self.thresh = 0.5
        self.arrayOfWeights = []
        self.numofclassify = 0




    def ROC_curve(self,y_true,y_prob):
        y_true = np.array([0 if x == -1 else 1 for x in y_true])
        thresholds = np.linspace(0, 1, 100)
        tpr_values = []
        fpr_values = []
        num_positive_cases = sum(y_true)
        num_negative_cases = len(y_true) - num_positive_cases
        for threshold in thresholds:
            y_pred = np.where(y_prob < threshold, 0, 1)
            tp = np.sum((y_true == 1) & (y_pred == 1))  # true positive
            fp = np.sum((y_true == 0) & (y_pred == 1))  # false positive
            tpr = tp / num_positive_cases  # create tpr score (recall)
            fpr = fp / num_negative_cases  # create fpr score (missing rate)
            # record the scores
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        plt.scatter(fpr_values, tpr_values, label='ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        best_threshold_index = np.argmax(np.array(tpr_values)-np.array(fpr_values))  # find the most successful threshold index
        best_threshold = thresholds[best_threshold_index]
        plt.scatter(fpr_values[best_threshold_index], tpr_values[best_threshold_index], marker='o', facecolors='green', s=40)
        plt.show()
        return best_threshold




    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        ones_column = np.ones((X.shape[0], 1), dtype=X.dtype)  # adding a '1' column to represent the 'b' in the module.
        X = np.hstack((ones_column, X))
        N = X.shape[0]

        w = np.random.randn(X.shape[1])
        for _ in range(self.iterations):
            predictions = (2 * self.sigmoid(np.dot(X, w))) - 1
            prob_dif = y - predictions
            gradient = -np.dot(X.T, prob_dif) / N
            if (np.linalg.norm(gradient) >= 0.001):
                w -= self.learning_rate * gradient
            else:
                break

        self.weights_ = w


    def predict(self, X, add_bias=True):
        if add_bias:  # if we already add bias in the other function we don't need to add another '1' column
            ones_column = np.ones((X.shape[0], 1), dtype=X.dtype)
            X = np.hstack((ones_column, X))

        res1 = self.predict_proba(X, False)
        res1 = np.where(res1 >= self.thresh, 1, -1)
        return res1

    def predict_proba(self, X, add_bias=True):
        if add_bias:  # if we already add bias in the other function we don't need to add another '1' column
            ones_column = np.ones((X.shape[0], 1), dtype=X.dtype)
            X = np.hstack((ones_column, X))

        return self.sigmoid(np.dot(X, self.weights_))

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

    def multiFit(self,X,y):
        self.numofclassify = len(set(y))
        for i in range(self.numofclassify):
            y_temp = pd.Series([1 if x == i else -1 for x in y])
            self.fit(X, y_temp)
            self.arrayOfWeights.append(self.weights_)

    def predictMulti(self,X):
        y_pred_proba = []
        for i in range(self.numofclassify):
            self.weights_ = self.arrayOfWeights[i]
            y_pred_proba.append(self.predict_proba(X))

        zipped = zip(*y_pred_proba)
        res = []
        for item in zipped:
            index_of_max = item.index(max(item))
            res.append(index_of_max)
        return res

    def scoreMulti(self, X, y):
        pred = self.predictMulti(X)
        compared = self.compare_lists(y, pred)
        return np.sum(compared) / len(y)




##these two function are another gradient decent algorithm but with less successful score.
    def gradient(self, w, X, y):
        N, n = X.shape  # N: number of samples, n: number of features
        gradient = np.zeros(n)
        for i in range(N):
            x_i = X[i]
            y_i = y[i]
            z_i = y_i * np.dot(w,x_i)
            gradient += (-y_i * x_i) / (self.sigmoid(-z_i))
        gradient = (1/N) * np.array(gradient)
        return np.array(gradient)

    def calc_gradient(self, w, X, y):
        a = -y*self.sigmoid(-y*(X@w))
        return np.dot(a,X)
    def gradient_descent(self, X, y):
        ones_column = np.ones((X.shape[0], 1), dtype=X.dtype)  # adding a '1' column to represent the 'b' in the module.
        X = np.hstack((ones_column, X))
        w = np.zeros(X.shape[1])
        # Gradient decent
        for i in range(self.iterations):
            g = self.calc_gradient(w,X,y)  # g stand for gradient
            if(np.linalg.norm(g) >= 0.005):#0.0085):
                w = w - self.learning_rate * g
            else:
                break
        self.weights_ = w

        ##gradient_descent