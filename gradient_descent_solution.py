from utils import x_values, y_values, plot_line_and_points, run_func_with_runtime


# For simplicity, script only have partial derivative of RSS wrt b and w
def get_gradient_at_b(x_values, y_values, w, b):
    diff = 0
    N = len(x_values)
    for i in range(N):
        y = y_values[i]
        x = x_values[i]
        diff += (y - ((w * x) + b))
    b_gradient = -2 / N * diff
    return b_gradient


def get_gradient_at_w(x_values, y_values, w, b):
    diff = 0
    N = len(x_values)
    for i in range(N):
        y = y_values[i]
        x = x_values[i]
        diff += x * (y - ((w * x) + b))
    w_gradient = -2 / N * diff
    return w_gradient


def gradient_step(x_values, y_values, w_current, b_current, learning_rate):
    w = w_current - (get_gradient_at_w(x_values, y_values, w_current, b_current) * learning_rate)
    b = b_current - (get_gradient_at_b(x_values, y_values, w_current, b_current) * learning_rate)
    return b, w


def main(iter_count=1000, learning_rate=0.02):
    # Gradient solution needs any point to start both for w and b,
    # they're assumed 5 here, because normal solution is close to 0
    b = 5
    w = 5
    for i in range(iter_count):
        b, w = gradient_step(x_values, y_values, w, b, learning_rate)
    plot_line_and_points(b, w, x_values, y_values)


if __name__ == '__main__':
    run_func_with_runtime(main)
