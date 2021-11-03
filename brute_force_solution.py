from utils import (
    plot_line_and_points,
    find_rss,
    plot_3d_points,
    xs as x_values,
    ys as y_values,
    get_execution_time_by_running)

import numpy as np


def main(print_rss=True, plot_changes=False):
    min_error = float('inf')
    min_w = None
    min_b = None

    # Lines change a lot that's why to render less plot
    # change_count will be used to print only 1 time per 13 change as below.
    change_count = 0
    for b in range(-100, 100, 1):
        rss_for_b = []
        for w in range(-100, 100, 1):
            rss = find_rss(w, b, x_values, y_values)
            rss_for_b.append(rss)
            if print_rss:
                print("RSS: ", rss)
            if rss < min_error:
                change_count += 1
                min_w = w
                min_b = b
                min_error = rss
                if change_count == 13 and plot_changes:
                    change_count = 0
                    plot_line_and_points(b, w, x_values, y_values)
    plot_line_and_points(min_b, min_w, x_values, y_values)
    print(min_b, min_w)
    return (min_b, min_w)

if __name__ == '__main__':
    get_execution_time_by_running(main, print_rss=False, plot_changes=False)
