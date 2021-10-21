import numpy as np
from utils import xs, ys, plot_line_and_points, get_execution_time_by_running


def find_b_with_closed_form(x_values, y_values, w):
    # When we derive RSS with respect to b
    # equation becomes as below
    n = len(x_values)
    sum_of_w_times_xs = sum(x_values) * w
    sum_of_ys = sum(y_values)
    b = (sum_of_ys - sum_of_w_times_xs) / n
    return b


def find_w_with_closed_form(x_values, y_values):
    # When we derive RSS with respect to (wrt) w
    # and use derivation of RSS wrt. b for simplification
    # equation becomes as below

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    n = len(x_values)
    sum_of_ys = sum(y_values)
    sum_of_xs = sum(x_values)
    sum_of_ys_product_xs = np.sum(x_values * y_values)
    w = ((sum_of_ys * sum_of_xs) - (sum_of_ys_product_xs * n)) / (sum_of_xs ** 2 - (n * np.sum(x_values * x_values)))
    return w


def main():
    w = find_w_with_closed_form(xs, ys)
    b = find_b_with_closed_form(xs, ys, w)
    plot_line_and_points(b, w, xs, ys)
    print(b, w)


if __name__ == '__main__':
    get_execution_time_by_running(main)