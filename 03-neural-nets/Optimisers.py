import numpy as np

class SGDOptimizer:
    def __init__(self, lr=0.1, rho=0.):
        """
            A class containing the Stochastic Gradient Descent algorithm
            lr: (float)
                the learning rate
            rho: (float)
                the value of the momentum
        """
        self.lr = lr
        self.rho = rho

        # initializing the vectors of forwards outputs
        self.outputs = []
        # initializing the vectors of backwards outputs (the gradients)
        self.grads = []

    def update_grads(self, grads, k):
        """
            Compute the changes to the gradients
            grads: (list)
                list of gradients of a certain layer
            k: (int)
                index of the layer
        """

        # momentum update
        if self.grads[k] != []:
            self.grads[k] = [self.rho * self.grads[k][i] + (1. - self.rho) * grads[i] for i in range(len(grads))]
        else:
            self.grads[k] = [(1. - self.rho) * grads[i] for i in range(len(grads))]

        self.grads[k] = [np.clip(grad, -0.5, 0.5) for grad in self.grads[k]]
        self.grads[k] = [grad / np.linalg.norm(grad) for grad in self.grads[k]]

        # return the final change to make the parameters
        return [-self.lr * grad for grad in self.grads[k]]

    def apply_gradients(self, layers, grad_loss):
        """
            Compute the gradients of the loss w.r.t each layers' parameters
            and update the parameters.
            layers: (list)
                list of layers (the inputs are in layers[0])
            grad_loss: (array-like)
                gradient of the loss w.r.t the final output
        """
        K = len(layers) - 1
        for k in range(K, -1, -1):
            l_grads, grad_loss = layers[k].compute_gradients(self.outputs[k], grad_loss)
            layers[k].update_params(self.update_grads(l_grads, k))


class RMSPropOptimizer:
    def __init__(self, lr=0.1, rho=0.):
        """
            A class containing the Stochastic Gradient Descent algorithm
            lr: (float)
                the learning rate
            rho: (float)
                the value of the momentum
        """
        self.lr = lr
        self.rho = rho

        # initializing the vectors of forwards outputs
        self.outputs = []
        # initializing the vectors of backwards outputs (the gradients)
        self.grads = []

    def update_grads(self, grads, k):
        """
            Compute the changes to the gradients
            grads: (list)
                list of gradients of a certain layer
            k: (int)
                index of the layer
        """
        # momentum update
        if self.grads[k] != []:
            self.grads[k] = [self.rho * self.grads[k][i] + (1. - self.rho) * grads[i] ** 2 for i in range(len(grads))]
        else:
            self.grads[k] = [(1. - self.rho) * grads[i] ** 2 for i in range(len(grads))]

        self.grads[k] = [np.clip(grad, -0.5, 0.5) for grad in self.grads[k]]
        self.grads[k] = [grad / np.linalg.norm(grad) for grad in self.grads[k]]

        # return the final change to make the parameters
        return [-self.lr * grads[i] * np.power(self.grads[k][i] + 1e-6, -0.5) for i in range(len(grads))]

    def apply_gradients(self, layers, grad_loss):
        """
            Compute the gradients of the loss w.r.t each layers' parameters
            and update the parameters.
            layers: (list)
                list of layers (the inputs are in layers[0])
            grad_loss: (array-like)
                gradient of the loss w.r.t the final output
        """
        K = len(layers) - 1
        for k in range(K, -1, -1):
            l_grads, grad_loss = layers[k].compute_gradients(self.outputs[k], grad_loss)
            layers[k].update_params(self.update_grads(l_grads, k))