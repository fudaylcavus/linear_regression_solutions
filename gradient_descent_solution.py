from utils import xs, ys, get_execution_time_by_running, plot_polynom_and_points
import numpy as np


def get_diff_for_single_x(x, ws, w_index, x_index):
    # w0 * zeroth power of x , w1 * first power of x + w2 * second power of x ...
    x_s = [pow(x, i) for i in range(len(ws))]  # [0, x1, x1**2, ...]
    x_s = np.array(x_s)
    ws = np.array(ws)
    diff = ys[x_index] - np.sum(x_s * ws)  # matrix multiplication and sum of the result
    return diff * x_s[w_index]


def get_total_diff(ws, w_index):
    total_diff = 0
    N = len(xs)
    for i in range(N):
        total_diff += get_diff_for_single_x(xs[i], ws, w_index, i)
    return total_diff


def get_gradient_at_w(ws, w_index):
    N = len(xs)
    gradient_w = (-2 / N) * get_total_diff(ws, w_index)
    return gradient_w


def gradient_step_for_w(ws, w_index, w_current, learning_rate):
    ws[w_index] = w_current - (get_gradient_at_w(ws, w_index) * learning_rate)


def main(iter_count=1000, learning_rate=0.03, degree=1):
    # degree + 1 cause of intercept (w0 or bias)
    ws = np.zeros(degree + 1, dtype=np.float64)
    for _ in range(iter_count):
        for i in range(len(ws)):
            gradient_step_for_w(ws, i, ws[i], learning_rate)
    plot_polynom_and_points(ws)


if __name__ == '__main__':
    get_execution_time_by_running(main)
