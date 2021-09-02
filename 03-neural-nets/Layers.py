import numpy as np
from Activations import *
class DenseLayer:
    def __init__(self, n_neurons, activation="sigmoid", l2_regul=0.0):
        """
            A Dense layer class (Fully-connected layer)
            n_neurons: (int)
                number of neurons in the layer
            activation: (string)
                activation function to use
        """
        self.N = n_neurons
        self.l2_regul = l2_regul
        if activation == "sigmoid":
            self.activ_func = Sigmoid
        elif activation == 'relu':
            self.activ_func = ReLU
        elif activation == "softmax":
            self.activ_func = Softmax
        else:
            self.activ_func = NoActivation

    def build(self, input_shape):
        """
            Initialize the layer by creating the parameters
            input_shape: (list or tuple or array)
                shape of the inputs. Must be of the form (batch_size, input_dim)
        """
        # self.W = 0.01 * np.random.randn(self.N, input_shape[1])
        # self.b = 0.01 * np.random.randn(self.N)

        # Glorot Uniform initialization
        l = np.sqrt(6. / (self.N + input_shape[1]))
        self.W = np.random.uniform(low=-l, high=l, size=(self.N, input_shape[1]))
        self.b = np.random.uniform(low=-l, high=l, size=self.N)

    def preactiv(self, inputs):
        """
            Computes the preactivation of the inputs
            inputs: (array-like), shape=(batch_size, input_dim)
                the inputs to the layer
        """
        return np.dot(inputs, self.W.T) + self.b

    def call(self, inputs):
        """
            Computes the preactivation and apply the activation function
            inputs: (array-like), shape=(batch_size, input_dim)
                the inputs to the layer
        """
        return self.activ_func.call(self.preactiv(inputs))

    def get_output_shape(self, input_shape):
        """
            Returns the output shapte of the layer
            input_shape: (list or tuple or array)
                shape of the inputs. Must be of the form (batch_size, input_dim)
        """
        out_shape = list(input_shape)
        out_shape[-1] = self.N
        return out_shape

    def compute_gradients(self, inputs, back_grads):
        """
            Computes the gradients of this layers' parameters
            inputs: (array-like), shape=(batch_size, input_dim)
                the inputs to the layer
            back_grads: (array-like), shape=(batch_size, self.N)
                gradients backpropagated from the next layer
        """
        m = inputs.shape[0]
        dzk_dqk = self.activ_func.grad(self.preactiv(inputs))
        back_grads = back_grads * dzk_dqk

        grad_w = np.dot(back_grads.T, inputs) / m + self.l2_regul * self.W
        grad_b = np.sum(back_grads, axis=0) / m

        back_grads = np.dot(back_grads, self.W)

        return [grad_w, grad_b], back_grads

    def update_params(self, grads):
        """
            Updates the parameters using the changes provided by the Optimizer class
            grads: (list)
                gradients of each parameter of this layer
        """
        self.W += grads[0]
        self.b += grads[1]