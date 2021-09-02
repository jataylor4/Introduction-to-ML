import numpy as np

class LinearSVM:
    def __init__(self, n_iter, lr, C):
        """
        A class for linear SVM model regression
            :param n_iter:          Number of iterations in training
            :param lr:              Learning rate
            :param C:           Regularisation weight
        """
        self.n_iter = n_iter
        self.lr = lr
        self.C = C

        self.losses = []
        self.grads = []

    def init_params(self, n_feats):
        """
        Function for initialising the parameters of the model
            :param n_feats:         Number of features in input data
        """
        self.w = np.zeros((1, n_feats))
        self.b = 0

    def _optimize(self, X, y):
        """
        Compute cost function and gradients for parameters including regularisation
            :param X:           Input data
            :param y:           Target datat
            :return grads:      Computed gradients for w,b
            :return cost:       Computed error of cost function
        """
        m = X.shape[0]

        dist = 1. - y * (np.dot(self.w, X.T) + self.b)
        dist[dist < 0] = 0
        # cost function
        cost = 0.5 * np.dot(self.w, self.w.T) + self.C * np.mean(dist)

        # computing the gradient
        dw = 0;
        db = 0
        for i, d in enumerate(np.squeeze(dist)):
            if d != 0:
                dw += self.w - self.C * np.squeeze(y)[i] * X[i]
                db += -self.C * np.squeeze(y)[i]
            else:
                dw += self.w
                db += 0

        dw *= (1. / m)
        db *= (1. / m)

        grads = {"dLdw": dw, "dLdb": db}

        return grads, cost

    def activation(self, X):
        """
        Compute the prediction rule
            :param X:       Input data
            :return:        Output of the prediction
        """
        return np.sign(np.dot(self.w, X.T) + self.b)

    def fit(self, X, y):
        """
        Training iteration loop to fit the line of best fit
            :param X:           Input data
            :param y:           Target data
        """
        assert len(np.unique(y)) == 2, "More than two labels in y %s" % (np.unique(y))
        labels = list(np.squeeze(np.unique(y)))
        remapped_labels = y
        if labels != [-1, 1]:
            remapped_labels = np.zeros_like(y)
            remapped_labels[y == np.unique(y)[0]] = 1
            remapped_labels[y == np.unique(y)[1]] = -1

        self.init_params(X.shape[1])

        for i in range(self.n_iter):
            grads, cost = self._optimize(X, remapped_labels)
            #
            dLdw = grads['dLdw']
            dLdb = grads['dLdb']

            # gradient descent
            self.w = self.w - self.lr * dLdw
            self.b = self.b - self.lr * dLdb

            self.losses.append(cost)
            self.grads.append(grads)
            # if (i % 10 == 0):
            #    print("Standard Logistic Regression: Iter {}, Cost {}".format(i, cost))

        print("Standard Logistic Regression: Iter {}, Cost {}".format(i, cost))

    def predict(self, X):
        """
            :param X:       Input data matrix
            :return:        The binary classification of the SVM  model
        """
        return self.activation(X)

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