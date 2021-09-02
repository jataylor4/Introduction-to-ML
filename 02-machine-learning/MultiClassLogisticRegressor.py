import numpy as np
import Utils

class MultiClassLogisticRegression:
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
        self.regul = regul
        self.r_lambda = r_lambda

        self.losses = []
        self.grads = []

    def init_params(self, n_feats, n_class):
        """
        Function for initialising the parameters of the model
            :param n_feats:         Number of features in input data
        """
        self.W = np.zeros((n_class, n_feats))
        self.b = np.zeros(n_class)

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
        cost = (-1 / m) * np.sum(y * np.log(probs))

        # computing the gradient
        dw = [];
        db = []
        for i in range(self.W.shape[0]):
            dw.append(-(1 / m) * (np.dot((- y[:, i] * probs[:, i] + y[:, i]), X)))

            if self.regul == 'l2':
                dw[-1] += 2 * self.r_lambda * self.W[i]  # adding l2 regularization
            elif self.regul == 'l1':
                dw[-1] += self.r_lambda * np.sign(self.W[i])  # adding l1 regularization

            db.append(-(1 / m) * np.sum(-y[:, i] * probs[:, i] + y[:, i]))

        grads = {"dLdw": np.array(dw), "dLdb": np.array(db)}

        return grads, cost

    def activation(self, X):
        """
        Compute the Sigmoid activation function
            :param X:       Input data
            :return:        Output of the sigmoid activation function
        """
        vect_of_exponents = np.exp(np.dot(X, self.W.T) + self.b)
        return vect_of_exponents/ np.tile(np.sum(vect_of_exponents, 1), (self.W.shape[0],1)).T

    def fit(self, X, y):
        """
        Training iteration loop to fit the line of best fit
            :param X:           Input data
            :param y:           Target data
        """
        one_hot_labels = y
        if y.ndim < 2:
            one_hot_labels = one_hot_encode(y)
        self.init_params(X.shape[1], y.shape[1])

        for i in range(self.n_iter):
            grads, cost = self._optimize(X, one_hot_labels)
            #
            dLdw = grads['dLdw']
            dLdb = grads['dLdb']

            # gradient descent
            self.W = self.W - self.lr * dLdw
            self.b = self.b - self.lr * dLdb

            self.losses.append(cost)
            self.grads.append(grads)

        print("Multiclass Logistic Regression: Iter {}, Cost {}".format(i, cost))

    def predict(self, X):
        """
            :param X:       Input data matrix
            :return:        The binary classification of the logistic regression model
        """
        activ = self.activation(X)
        return np.argmax(activ, 1)

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