import numpy as np

class BinaryLogisticRegression:
    def __init__(self, n_iter, lr, regul='l2', r_lambda=0.):
        """
        A class for binary logistic regression
            :param n_iter:          Number of iterations in training
            :param lr:              Learning rate
            :param regul:           Regularisation ('l1' or 'l2')
            :param r_lambda:        Value for lambda
        """
        self.n_iter = n_iter
        self.lr = lr
        self.r_lambda = r_lambda
        self.regul = regul

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
        probs = self.activation(X)
        cost = (-1 / m) * (np.sum((y * np.log(probs)) + ((1 - y) * (np.log(1 - probs)))))

        # computing the gradient
        dw = (1 / m) * (np.dot((probs - y), X))

        if self.regul == "l2":
            dw += 2 * self.r_lambda * self.w  # adding l2 regularization
        elif self.regul == 'l1':
            dw + self.r_lambda * np.sign(self.w)  # adding l1 regularization

        db = (1 / m) * (np.sum(probs - y))

        grads = {"dLdw": dw, "dLdb": db}

        return grads, cost

    def activation(self, X):
        """
        Compute the Sigmoid activation function
            :param X:       Input data
            :return:        Output of the sigmoid activation function
        """
        return 1. / (1. + np.exp(-np.dot(self.w, X.T) - self.b))

    def fit(self, X, y):
        """
        Training iteration loop to fit the line of best fit
            :param X:           Input data
            :param y:           Target data
        """
        self.init_params(X.shape[1])

        for i in range(self.n_iter):
            grads, cost = self._optimize(X, y)

            dLdw = grads['dLdw']
            dLdb = grads['dLdb']

            # gradient descent
            self.w = self.w - self.lr * dLdw
            self.b = self.b - self.lr * dLdb

            self.losses.append(cost)
            self.grads.append(grads)

        print(f"Standard Logistic Regression: Iter {i}, Cost {cost}")

    def predict(self, X):
        """
            :param X:       Input data matrix
            :return:        The binary classification of the logistic regression model
        """
        activ = self.activation(X)
        return activ >= 0.5

    def predict_proba(self, X):
        """
        Predicting the probabilities by applying the activation function
            :param X:           Input matrix
            :return:            Return the activation function output
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