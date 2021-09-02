import numpy as np

class NN:
    def __init__(self, input_shape):
        """
            A generic Neural Network class
            Params
            input_shape: (list or tuple or array)
                shape of the inputs. Must be of the form (batch_size, input_dim)
        """
        self.input_shape = input_shape
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile_model(self, loss, optimizer):
        self.loss = loss
        self.opt = optimizer
        self.n_layers = len(self.layers)

        input_shape = self.input_shape
        for l in self.layers:
            l.build(input_shape)
            input_shape = l.get_output_shape(input_shape)
            self.opt.outputs.append([])
            self.opt.grads.append([])

        self.opt.outputs.append([])
        self.opt.grads.append([])

    def forward(self, X):
        self.opt.outputs[0] = X
        for i in range(1, self.n_layers + 1):
            self.opt.outputs[i] = self.layers[i - 1].call(self.opt.outputs[i - 1])
        return self.opt.outputs[-1]

    def backward(self, y, zK):
        grad_loss = self.loss.grad(y, zK)
        self.opt.apply_gradients(self.layers, grad_loss)
        return grad_loss

    def fit(self, X, y, n_iter=100, batch_size=32, verbose=0):
        """
            Training the model
            X: (np-array) shape=(n_samples x n_feats)
                data matrix
            y: (np-array) shape=( n_samples)
                targets
            n_iter: (int)
                number of iterations
        """
        self.losses = []
        for e in range(n_iter):
            batch_loss = 0.
            for i in range(len(X) // batch_size):
                output = self.forward(X[i:i + batch_size])
                batch_loss += self.loss.call(y[i:i + batch_size], output)
                self.backward(y[i:i + batch_size], output)
            self.losses.append(batch_loss / (len(X) // batch_size))
            if verbose:
                print('Epoch [{}/{}], loss {:.8f}'.format(e, n_iter, self.losses[-1]))

    def predict(self, X):
        """
            Predicting the discrete labels
            X: (np-array) shape=(n_samples x n_feats)
                data matrix
        """
        return self.forward(X)