import pandas as pd
import numpy as np
import math
from LogisticRegression import LogisticRegression
from sklearn import datasets
from sklearn.utils import shuffle

if __name__ == "__main__":

    iris = datasets.load_iris()
    iris_data_shuffled = shuffle(iris.data, random_state=42)
    iris_target_shuffled = shuffle(iris.target, random_state=42)



    X_train, X_test, y_train, y_test = LogisticRegression.train_test_split(pd.DataFrame(iris_data_shuffled), pd.Series(iris_target_shuffled))  # split the dataset

    logistic_regression = LogisticRegression()
    logistic_regression.multiFit(X_train, y_train)
    print("The weights are: ",logistic_regression.arrayOfWeights)
    print("The score is: ", logistic_regression.scoreMulti(X_test, y_test))





