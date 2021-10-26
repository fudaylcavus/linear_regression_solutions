import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Polynomial dots generation
trX = np.linspace(-1, 1, 101)
num_coeffs = 6
trY_coeffs = [1, 2, 3, 4, 5, 6]
trY = 0
for i in range(num_coeffs):
    trY += trY_coeffs[i] * np.power(trX, i)
    trY += np.random.randn(*trX.shape) * .5

plt.scatter(trX, trY)
plt.show()
xs = trX
ys = trY

# # xs, x_test, ys, y_test = train_test_split(trX, trY)


def find_y_for_x(x_index, coeffs):
    x_powed = [pow(xs[x_index], i) for i in range(len(coeffs))]
    coeffs = np.array(coeffs)
    return np.sum(x_powed * coeffs)


def plot_polynom_and_points(coeffs):
    fx = []
    for i in range(len(xs)):
        fx.append(find_y_for_x(i, coeffs))
    plt.plot(xs, fx)
    plt.plot(xs, ys, "o")
    plt.show()


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


def get_execution_time_by_running(func, **kwargs):
    start = time.time()
    func(**kwargs)
    end = time.time()
    print('Runtime: ', round((end - start), 2), 's')
