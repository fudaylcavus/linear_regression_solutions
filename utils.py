import time
import matplotlib.pyplot as plt
import numpy as np

# Predefined linearish points for tests
y_values = [1.3, 2.1, 3.6, 4, 5.9, 5.6, 7, 9, 8.4]
x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def plot_line_and_points(b, w, x_values, y_values):
    # since 'plot' method automatically connects dots to make a line
    # selecting minimum and maximum point would include all other points
    # cause of it's linear
    min_x = min(x_values)
    max_x = max(x_values)
    xs = [min_x, max_x]
    ys = [b + w * x for x in xs]
    plt.plot(xs, ys)
    plt.plot(x_values, y_values, 'o')
    plt.show()


def plot_3d_points(x_values, y_values, z_values):
    Z = np.array(z_values)
    X, Y = np.meshgrid(x_values, y_values)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_wireframe(X, Y, Z)
    plt.show()


def find_rss(w, b, x_real, y_real):
    rss = 0
    for x in range(len(x_real)):
        square_of_residue = (y_real[x] - (b + w * x_real[x])) ** 2
        rss += square_of_residue
    return rss


def run_func_with_runtime(func, **kwargs):
    start = time.time()
    func(**kwargs)
    end = time.time()
    print('Runtime: ', round((end - start), 2), 's')
