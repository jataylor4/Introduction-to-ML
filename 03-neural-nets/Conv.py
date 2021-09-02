from scipy.signal import convolve

class Conv2DLayer:
    def __init__(self, n_filters, size=(3, 3), padding="valid", activation="sigmoid",
                 initializer="xavier", l2_regul=0.0):
        """
            A Conv layer class

            Params
            -------
            n_filters: (int)
                number of kernels in the layer
            size: (list or tuple or array)
                height and width of the kernels
            padding: (string)
                type of padding (same or valid)
            activation: (string)
                activation function to use
        """
        self.N = n_filters
        self.k_size = size
        self.l2_regul = l2_regul
        self.initializer = initializer
        self.padding = padding

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

            Params
            -------
            input_shape: (list or tuple or array)
                shape of the inputs. Must be of the form (batch_size, input_dim)
        """
        k_shape = (self.N, self.k_size[0], self.k_size[1], input_shape[-1])
        if self.initializer == "rand":
            self.W = 0.01 * np.random.randn(*k_shape)
            self.b = 0.01 * np.random.randn(self.N)
        else:
            # Glorot Uniform initialization
            l = np.sqrt(6. / (self.N + input_shape[1]))
            self.W = np.random.uniform(low=-l, high=l, size=k_shape)
            self.b = np.random.uniform(low=-l, high=l, size=self.N)

        self.n_pad = (0, 0) if self.padding == "valid" else (self.k_size[0] // 2, self.k_size[1] // 2)

    def preactiv(self, inputs):
        """
            Computes the preactivation of the inputs

            Params
            -------
            inputs: (array-like), shape=(batch_size, input_dim)
                the inputs to the layer
        """
        return np.array([np.array([convolve(inputs[i], self.W[j, ::-1, ::-1, :], mode=self.padding)
                                   for j in range(self.N)]).sum(-1).transpose(1, 2, 0)
                         for i in range(len(inputs))])

    def call(self, inputs):
        """
            Computes the preactivation and apply the activation function

            Params
            -------
            inputs: (array-like), shape=(batch_size, input_dim)
                the inputs to the layer
        """
        return self.activ_func.call(self.preactiv(inputs))

    def get_output_shape(self, input_shape):
        """
            Returns the output shapte of the layer

            Params
            -------
            input_shape: (list or tuple or array)
                shape of the inputs. Must be of the form (batch_size, height, width, n_channels)
        """
        out_shape = list(input_shape)

        out_shape[-1] = self.N
        out_shape[1] = (input_shape[1] + 2 * self.n_pad[0] - self.k_size[0]) + 1
        out_shape[2] = (input_shape[2] + 2 * self.n_pad[1] - self.k_size[1]) + 1
        return out_shape

    def compute_gradients(self, inputs, back_grads):
        """
            Computes the gradients of this layers' parameters

            Params
            -------
            inputs: (array-like), shape=(batch_size, input_dim)
                the inputs to the layer
            back_grads: (array-like), shape=(batch_size, self.N)
                gradients backpropagated from the next layer
        """
        m = inputs.shape[0]

        inputs = np.pad(inputs, ((0,), (0,), (self.n_pad[0],), (self.n_pad[1],)), 'constant')
        back_grads = back_grads * self.activ_func.grad(back_grads)

        pg_1, pg_2 = back_grads.shape[1], back_grads.shape[2]
        grad_w = np.array([[[
            (inputs[:, a:a + pg_1, b:b + pg_2, :] * back_grads[:, :, :, f:f + 1]).sum((0, 1, 2))
            for b in range(self.k_size[1])]
            for a in range(self.k_size[0])]
            for f in range(self.N)])

        grad_b = 0

        back_grads = np.array([np.array([convolve(back_grads[i], self.W[j, ::-1, ::-1, :], mode="same")
                                         for j in range(self.N)]).sum(-1).transpose(1, 2, 0)
                               for i in range(len(inputs))])

        return [grad_w, grad_b], back_grads

    def update_params(self, grads):
        """
            Updates the parameters using the changes provided by the Optimizer class

            Params
            -------
            grads: (list)
                gradients of each parameter of this layer
        """
        self.W += grads[0]
        self.b += grads[1]


class FlattenLayer:
    def __init__(self):
        """
            A Flatten layer class

            Params
            -------
        """
        pass

    def build(self, input_shape):
        """
            Initialize the layer by creating the parameters

            **Not used by the FlattenLayer**

            Params
            -------
            input_shape: (list or tuple or array)
                shape of the inputs. Must be of the form (batch_size, input_dim)
        """
        pass

    def call(self, inputs):
        """
            Computes the preactivation and apply the activation function

            Params
            -------
            inputs: (array-like), shape=(batch_size, input_dim)
                the inputs to the layer
        """
        return inputs.reshape(inputs.shape[0], -1)

    def get_output_shape(self, input_shape):
        """
            Returns the output shapte of the layer

            Params
            -------
            input_shape: (list or tuple or array)
                shape of the inputs. Must be of the form (batch_size, height, width, n_channels)
        """
        in_shape = list(input_shape)
        out_shape = [in_shape[0], np.prod(in_shape[1:])]
        return out_shape

    def compute_gradients(self, inputs, back_grads):
        """
            Computes the gradients of this layers' parameters

            Params
            -------
            inputs: (array-like), shape=(batch_size, input_dim)
                the inputs to the layer
            back_grads: (array-like), shape=(batch_size, self.N)
                gradients backpropagated from the next layer
        """
        return [0], back_grads.reshape(inputs.shape)

    def update_params(self, grads):
        """
            Updates the parameters using the changes provided by the Optimizer class

            **Not used by the FlattenLayer**

            Params
            -------
            grads: (list)
                gradients of each parameter of this layer
        """
        pass