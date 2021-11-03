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
        self.sigma = None
        self.N = None

    def z_score_fit(self, xs: np.ndarray):
        self.mean = xs.mean()
        self.sigma = np.std(xs)
        return self

    def z_score_transform(self, xs: np.ndarray):
        return (xs - self.mean) / self.sigma

    def score(self, xs, ys):
        # 1 - (rss/tss) gives the accuracy
        xs = self.z_score_transform(xs)
        y_predict = np.dot(self.coef_, xs.T) + self.intercept_
        RSS = ((ys - y_predict) ** 2).sum()
        TSS = ((ys - ys.mean()) ** 2).sum()
        return 1 - (RSS / TSS)

    def predict(self, xs):
        xs = self.z_score_transform(xs)
        return xs.dot(self.coef_) + self.intercept_


class SimpleRegression(Regression):
    def __init__(self, epoch, learning_rate, degree):
        super().__init__(epoch, learning_rate, degree)

    def fit(self, xs, ys):
        # degree + 1 cause of intercept (w0 or bias)
        ws = np.zeros(self.degree + 1, dtype=float)

        # Cause of it's polynomial for each X value we will use x to the power of [0, 1, ,2, 3, ,5.. degree+1]
        # with respect to index of W
        power_indexes = [np.repeat(i, len(xs)) for i in range(len(ws))]

        for _ in range(self.epoch):
            y_predict = np.dot(ws, np.power(xs, power_indexes))
            gradient = (-2 / len(xs)) * np.dot((ys - y_predict), np.power(xs, power_indexes).transpose())
            ws = ws - (gradient * self.learning_rate)
        self.coef_ = ws[1:]
        self.intercept_ = ws[0]


class MultipleRegression(Regression):
    def __init__(self, epoch, learning_rate, penalty):
        # For now, hardcoded degree to 1 to only calculate multiple linear regression
        super().__init__(epoch, learning_rate, 1, penalty)

    def fit(self, x_values, ys, how='ridge'):
        print(x_values.shape)
        self.N, self.degree = x_values.shape
        ws = np.zeros(self.degree + 1, dtype=float)
        xs = self.z_score_fit(x_values).z_score_transform(x_values)

        # adding every first indexes a 1 for intercept (w0 or b)
        xs = np.insert(xs, 0, 1, axis=1)
        ys = ys.reshape(-1)

        for _ in range(self.epoch):
            y_predict = xs.dot(ws)
            loss = (ys - y_predict)
            #
            gradient = (-2 * (xs.T.dot(loss) + (2 * ws * self.penalty))) / self.N
            ws = ws - (gradient * self.learning_rate)
        self.coef_ = ws[1:]
        self.intercept_ = ws[0]

    def plot_cost_by_epoch(self):
        plt.plot(self.cost_log)
        plt.xlabel('Epoch')
        plt.ylabel('RSS')
        plt.title('class')
        plt.show()


if __name__ == '__main__':
    model = MultipleRegression(2000, 0.03, 1)
    model.fit(x_train, y_train)
    print(model.coef_, model.intercept_)

    # Printing Scores
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print("Train Score: ", round(train_score, 2))
    print("Test Score: ", round(test_score, 2))

    # To clearly see y_predict and y_real with linear plot
    plt.plot(range(20000), range(20000), c='black')
    y_predict = model.predict(x_train)
    plt.scatter(y_train, y_predict, alpha=0.3)
    plt.xlabel('Y Real')
    plt.ylabel('Y Prediction')
    plt.show()
