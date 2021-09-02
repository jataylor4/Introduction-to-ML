import numpy as np

class Perceptron:
    def __init__(self, n_iter, lr=1):
        """
        A class for binary logistic regression
            :param n_iter:          Number of iterations in training
            :param lr:              Learning rate
        """
        self.n_iter = n_iter
        self.lr = lr

        self.losses = []
        self.grads = []

    def init_params(self, n_feats):
        """
        Function for initialising the parameters of the model
            :param n_feats:         Number of features in input data
        """
        self.w = np.zeros((1, n_feats))
        self.b = 0.

    def _optimize(self, X, y):
        """
        Compute cost function and gradients for parameters including regularisation
            :param X:           Input data
            :param y:           Target datat
            :return grads:      Computed gradients for w,b
            :return cost:       Computed error of cost function
        """
        m = X.shape[0]

        # cost function
        preds = np.squeeze(self.activation(X))
        distances = np.squeeze(preds != y)
        cost = np.mean(distances)

        # computing the rules (vectorized version)
        dw = (1 / m) * np.dot(preds - y, X)
        db = (1 / m) * np.sum(preds - y)

        grads = {"dLdw": dw, "dLdb": db}

        return grads, cost

    def activation(self, X):
        """
        Compute the activation function
            :param X:       Input data
            :return:        Output of the sigmoid activation function
        """
        return np.dot(self.w, X.T) + self.b > 0

    def fit(self, X, y):
        self.init_params(X.shape[1])

        for i in range(self.n_iter):
            grads, cost = self._optimize(X, y)
            #
            dLdw = grads['dLdw']
            dLdb = grads['dLdb']

            # gradient descent
            self.w = self.w - self.lr * dLdw
            self.b = self.b - self.lr * dLdb

            self.losses.append(cost)
            self.grads.append(grads)
            # if (i % 10 == 0):
            #    print("Standard Perceptron: Iter {}, Cost {}".format(i, cost))

        print("Standard Perceptron: Iter {}, Cost {}".format(i, cost))

    def predict(self, X):
        """
            :param X:       Input data matrix
            :return:        The binary classification of the logistic regression model
        """
        return self.activation(X)


    def score(self, X, y):
        """
        Compute the accuracy of the model
        :param X:           Input data matrix
        :param y:           Target data
        :return:            Accuracy of the model
        """
        pred = self.predict(X)
        return (pred == y).mean()