"""
Owner : Fudayl Ikbal Cavus 
Project : Linear Regression 
Data Set: StreetEasy Rental Real Estate Market Reports
Data Set Source: https://github.com/Codecademy/datasets/tree/master/streeteasy
Date : 03 November 2021
Time : 16:22
Copyright (C) belong to ComVIS Lab/NET
"""

from utils import (
    x_train,
    x_test,
    y_train,
    y_test,
)
from matplotlib import pyplot as plt
import numpy as np


class Regression:
    def __init__(self, epoch, learning_rate, degree, penalty=None):
        self.epoch = epoch
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.degree = degree
        self.coef_ = None
        self.intercept_ = None
        self.mean = None
        self.std = None
        self.N = None

    def z_score_fit(self, xs: np.ndarray):
        self.mean = xs.mean()
        self.std = np.std(xs)
        return self

    def z_score_transform(self, xs: np.ndarray):
        return (xs - self.mean) / self.std

    def score(self, xs, ys):
        # 1 - (rss/tss) gives the accuracy
        y_predict = self.predict(xs) 
        RSS = ((ys - y_predict) ** 2).sum()
        print(RSS)
        TSS = ((ys - ys.mean()) ** 2).sum()
        return 1 - (RSS / TSS)

    def predict(self, xs):
        xs, ys = self.validate(xs)
        xs = self.z_score_transform(xs)
        return xs.dot(self.coef_) + self.intercept_
    
    def validate(self, xs, ys=None):
        if type(xs) != np.ndarray:
            xs = np.array(xs, dtype=float)
        if type(ys) != np.ndarray:
            ys = np.array(ys, dtype=float)
        if xs.ndim == 1:
            xs = xs.reshape(-1, 1)
        return xs, ys



class LinearRegression(Regression):
    def __init__(self, epoch=1000, learning_rate=0.03, penalty=0):
        super().__init__(epoch, learning_rate, 1, penalty)

    def fit(self, x_values, y_values):
        x_values, y_values = self.validate(x_values, y_values)
        self.N, self.degree = x_values.shape
        ws = np.zeros(self.degree + 1, dtype=float)
        xs = self.z_score_fit(x_values).z_score_transform(x_values)

        # adding every first indexes a 1 for intercept (w0 or b)
        xs = np.insert(xs, 0, 1, axis=1)

        for _ in range(self.epoch):
            y_predict = xs.dot(ws)
            loss = (y_values - y_predict)
            #Ridge Regression
            gradient = (-2 * (xs.T.dot(loss) + (2 * ws * self.penalty))) / self.N
            ws = ws - (gradient * self.learning_rate)
        self.coef_ = ws[1:]
        self.intercept_ = ws[0]


if __name__ == '__main__':
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(model.coef_, model.intercept_)

    # Printing Scores
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print("Train Score: ", round(train_score, 2))
    print("Test Score: ", round(test_score, 2))

    # To clearly see y_predict and y_real with linear plot
    plt.plot(range(5), range(5), c='black')
    y_predict = model.predict(x_train)
    plt.scatter(y_train, y_predict, alpha=0.3)
    plt.xlabel('Y Real')
    plt.ylabel('Y Prediction')
    plt.show()
