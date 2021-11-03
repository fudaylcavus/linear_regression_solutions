import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Polynomial dots generation
trX = np.linspace(-1, 1, 101)
num_coeffs = 1
trY_coeffs = [ 2]
trY = 0
for i in range(num_coeffs):
    trY += trY_coeffs[i] * np.power(trX, i)
    trY += np.random.randn(*trX.shape) * .5

plt.scatter(trX, trY)
plt.show()
xs = trX
ys = trY

house_data = pd.read_csv('streeteasy.csv')
# x_all = house_data[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway']]
x_all = house_data[['size_sqft', 'bedrooms', 'bathrooms']]
y_all = house_data.rent  # thing we want to predict

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, train_size=50, random_state=0)
x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
# x_train = xs.reshape(-1, 1)
# y_train = ys.reshape(-1, 1)
# x_train = x_train[:5]
# y_train = y_train[:5]


def plot_polynom_and_points(coeffs, xs, ys):
    plt.plot(xs, xs.dot(coeffs), c='black')
    print(xs[:, 1])
    print(ys)
    plt.scatter(xs[:, 1], ys)

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


def plot_3d_points(x_values, y_values, z_values, z_prediction):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_values, y_values, z_values, c='red')

    ax.plot_trisurf(x_values, y_values, z_prediction, antialiased=True, linewidth=0)
    ax.view_init(10, 40)
    plt.show()


def find_rss(coeffs, x_real, y_real):
    cost = np.sum(((x_real.dot(coeffs) - y_real) ** 2) / (2 * len(y_real)))
    return cost
    # rss = ((y_real - (np.dot(coeffs,  x_real.T) + intercept))**2).sum()
    # return rss


def get_execution_time_by_running(func, **kwargs):
    start = time.time()
    output = func(**kwargs)
    end = time.time()
    print('Runtime: ', round((end - start), 2), 's')
    return output
