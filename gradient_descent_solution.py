from utils import xs, ys, get_execution_time_by_running, plot_polynom_and_points
import numpy as np


def main(iter_count=400, learning_rate=0.1, degree=1):
    # degree + 1 cause of intercept (w0 or bias)
    ws = np.zeros(degree + 1, dtype=float)

    # Cause of it's polynomial for each X value we will use x to the power of [0, 1, ,2, 3, ,5.. degree+1]
    # with respect to index of W
    power_indexes = [np.repeat(i, len(xs)) for i in range(len(ws))]

    for _ in range(iter_count):
        y_predict = np.dot(ws, np.power(xs, power_indexes))
        gradient = (-2 / len(xs)) * np.dot((ys - y_predict), np.power(xs, power_indexes).transpose())
        ws = ws - (gradient * learning_rate)
    plot_polynom_and_points(ws)
    print(ws)


if __name__ == '__main__':
    get_execution_time_by_running(main, degree=27)
