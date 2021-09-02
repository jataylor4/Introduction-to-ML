import numpy as np
import matplotlib.pyplot as plt

def least_squares(x, y):
    """
        Simple implementation of the least squares algorithm
            @return A - transposed input matrix
            @return w - weight (coefficient)
            @return b - bias (intercept)
    """
    a = np.vstack([x, np.ones(len(x))]).T # Concatenates arrays vertically with a array of 1s, and transposes
    w, b = np.linalg.lstsq(a, y, rcond=None)[0] # Computes a vector which approximately solves a*x=b
    return a, w, b

def gradient_descent(x, y, it):
    """
        Simple implementation of the Gradient Descent optimisation with LSE
            @return w - weight (coefficient)
            @return b - bias (intercept)
    """
    lr = 0.1 # Learning rate for updating the weights
    w = 1 # Initialise the weight vector
    b = 0 # Initialise the bias vector

    for i in range(it):

        # Initialising the gradient wrt w and b
        grad_w = 0
        grad_b = 0

        # Looping through the dataset
        for x_i, y_i in zip(x, y):
            grad_w += x_i * (x_i * w + b - y_i)
            grad_b += x_i * w + b - y_i

        # Gradient descent steps
        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b


def prepare_data():
    """
        Initialise the data arrays
          @return x - input data
          @return y - target data
    """
    x = np.array([0, 1, 2, 3])
    y = np.array([-1, 0.2, 0.9, 2.1])
    return x, y

def plot_results(x, y, w, b, title):
    """ Plot the line of best fit from the parameters """
    plt.title("Line of best fit: Least-square method")
    plt.plot(x, y, 'o', label='Input data', markersize=10)
    plt.plot(x, w*x + b, 'r', label='Fit')
    plt.legend()
    plt.savefig(title)
    plt.close()

if __name__ == "__main__":

    # Initialise data
    i_d, t_d = prepare_data()
    print(f"Input data:\n{i_d}\nTarget data:\n{t_d}\n-------------------\n")

    # Compute solution using least squares
    ans, weight, bias = least_squares(i_d, t_d)
    print(f"Least Squares Method:\n-----------------\n")
    print(f"The data matrix:\n{ans}\n")
    print(f"w = {weight},\t b = {bias}\n\n\n")

    # Plot solution 1
    plot_results(i_d, t_d, weight, bias, "least-squares")

    # Compute solution using gradient descent with 20 iterations
    weight, bias = gradient_descent(i_d, t_d, 20)
    print(f"Gradient Descent Optimisation:\n-----------------\n")
    print(f"w = {weight},\t b = {bias}\n")

    # Plot solution 2
    plot_results(i_d, t_d, weight, bias, "least-squares-gd")

