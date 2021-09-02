import numpy as np

class Sigmoid:
    @staticmethod
    def call(x):
        """
            Compute a sigmoid activation on the input
            A point-wise operation is performed for array-like inputs

            Params
            -------
            x: (array-like, or number)
                inputs

            Returns
            -------
            d: (same shape with the inputs)
        """
        return (1. / (1. + np.exp(-x))).clip(10e-8, 1 - 10e-8)

    @staticmethod
    def grad(x):
        """
            Compute a gradient of the sigmoid activation w.r.t the the input
            A point-wise operation is performed for array-like inputs

            Params
            -------
            x: (array-like, or number)
                inputs

            Returns
            -------
            d: (same shape with the inputs)
        """
        a = Sigmoid.call(x)
        return a * (1. - a)


class ReLU:
    @staticmethod
    def call(x):
        """
            Compute a ReLU activation on the input
            A point-wise operation is performed for array-like inputs

            Params
            -------
            x: (array-like, or number)
                inputs

            Returns
            -------
            d: (same shape with the inputs)
        """
        return np.maximum(0, x)

    @staticmethod
    def grad(x):
        """
            Compute a gradient of the ReLU w.r.t the inputs
            A point-wise operation is performed for array-like inputs

            Params
            -------
            x: (array-like, or number)
                inputs

            Returns
            -------
            d: (same shape with the inputs)
        """
        dx = np.ones_like(x)
        dx[x <= 0] = 0
        return dx


class Softmax:
    @staticmethod
    def call(x):
        """
            Compute a Softmax activation on the input
            A point-wise operation is performed for array-like inputs

            Params
            -------
            x: (array-like, or number)
                inputs

            Returns
            -------
            d: (same shape with the inputs)
        """
        v = np.exp(x)
        return (v / v.sum(1)[:, None]).clip(10e-8, 1 - 10e-8)

    @staticmethod
    def grad(x):
        """
            Compute a gradient of the softmax activation w.r.t the the input
            A point-wise operation is performed for array-like inputs

            Params
            -------
            x: (array-like, or number)
                inputs

            Returns
            -------
            d: (same shape with the inputs)
        """
        v = Softmax.call(x)
        return v - v ** 2


class NoActivation:
    @staticmethod
    def call(x):
        return x

    @staticmethod
    def grad(x):
        return x