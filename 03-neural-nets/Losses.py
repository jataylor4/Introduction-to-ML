import numpy as np

class MeanSquareLoss:
    @staticmethod
    def call(g, p):
        """
            Compute the MSE between two arrays of the same size
            g: (array-like)
               ground-truths
            p: (array-like)
                predictions

            Returns
            d: (real number) the MSE value

        """
        g = g.reshape(p.shape)
        return (.5 / len(p)) * np.sum((g - p) ** 2)

    @staticmethod
    def grad(g, p):
        """
            Compute the GRADIENT of the MSE w.r.t the input ''p''
            g: (array-like)
               ground-truths
            p: (array-like)
                predictions
            Returns
            d: (array-like) the gradient

        """
        g = g.reshape(p.shape)
        return (1. / len(p)) * (p - g)


class BinaryCrossEntropyLoss:
    @staticmethod
    def call(g, p):
        """
            compute the binary cross-entropy between two arrays of the same size
            g: (array-like)
               ground-truths
            p: (array-like)
                predictions
            Returns
            d: (real number) the binary CE

        """
        g = g.reshape(p.shape)
        return -(1. / len(p)) * np.sum(g * np.log(p) + (1. - g) * np.log(1. - p))

    @staticmethod
    def grad(g, p):
        """
            compute the Gradient of the BCE w.r.t the parameter ''p''
            g: (array-like)
               ground-truths
            p: (array-like)
                predictions
            Returns
            d: (array-like) the gradient

        """
        g = g.reshape(p.shape)
        return - (np.divide(g, np.maximum(p, 10e-5)) - np.divide(1. - g, 1. - p))


class CategoricalCrossEntropy:
    @staticmethod
    def call(g, p):
        g = g.reshape(p.shape)
        return (-1 / len(p)) * np.sum(g * np.log(p))

    @staticmethod
    def grad(g, p):
        g = g.reshape(p.shape)
        return (-1 / len(p)) * np.divide(g, p)