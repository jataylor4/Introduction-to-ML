import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_blobs, make_circles, make_moons, fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import Activations
import Layers
import Losses
import Model
import Optimisers
from Perceptron import Perceptron
from Utils import *
from Conv import *

def activation_functions():
    sigmoid = lambda t: 1. / (1. + np.exp(-t))
    tanh = lambda t: (np.exp(2 * t) - 1.) / (np.exp(2 * t) + 1.)
    relu = lambda t: np.maximum(0, t)
    prelu = lambda t, a: np.maximum(0, t) + a * np.minimum(0, t)
    elu = lambda t, a: np.maximum(0, t) + a * np.minimum(0, np.exp(t) - 1)

    xs = np.linspace(-5, 5, 100)

    plt.figure(figsize=(10, 10))

    plt.subplot(221)
    plt.plot(xs, sigmoid(xs))
    plt.title('Sigmoid')

    plt.subplot(222)
    plt.plot(xs, relu(xs))
    plt.title('ReLU')

    plt.subplot(223)
    plt.plot(xs, prelu(xs, 0.1))
    plt.title('PReLU')

    plt.subplot(224)
    plt.plot(xs, elu(xs, 0.1))
    plt.title('ELU')
    plt.savefig("figures/activation functions")

    plt.close()

def perceptron_training():

    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                               random_state=0, n_clusters_per_class=2, class_sep=2.0)

    my_clf = Perceptron(n_iter=100, lr=1.)
    my_clf.fit(X, y)
    rand_points = np.random.randn(10, 2)
    print_decision(X, y,  my_clf, "perceptron-boundary")
    print_decision(rand_points, np.ones(10), my_clf, "perceptron-boundary-random")

def ann_training():

    X, y = make_moons(noise=0.3, random_state=0)

    my_mlp = Model.NN(X.shape)
    my_mlp.add_layer(Layers.DenseLayer(32, activation="relu"))
    my_mlp.add_layer(Layers.DenseLayer(64, activation="relu"))
    my_mlp.add_layer(Layers.DenseLayer(128, activation="relu"))
    my_mlp.add_layer(Layers.DenseLayer(1, activation="sigmoid"))
    my_mlp.compile_model(Losses.BinaryCrossEntropyLoss, Optimisers.RMSPropOptimizer(lr=.01, rho=0.5))

    my_mlp.fit(X, y, 300, batch_size=len(X))
    print("score = ", np.mean(np.squeeze(my_mlp.predict(X) >= 0.5).astype(float) == y))

    plt.plot(range(len(my_mlp.losses)), my_mlp.losses);
    plt.savefig("figures/ann_moon_loss")
    print_decision(X, y, my_mlp, "ANN_decision_bondary")

def multi_class_ann():
    X_mclr, y_mclr = make_classification(n_samples=200, n_classes=4, n_features=2, n_redundant=0, n_informative=2,
                                         random_state=0, n_clusters_per_class=1, class_sep=1.5)

    # convert the targets to one-hot encoded vectors
    y_mclr = one_hot_encode(y_mclr)

    # design a multi-output network and compile it
    my_mlp_mc = Model.NN(X_mclr.shape)
    my_mlp_mc.add_layer(Layers.DenseLayer(32, activation="relu"))
    my_mlp_mc.add_layer(Layers.DenseLayer(64, activation="relu"))
    my_mlp_mc.add_layer(Layers.DenseLayer(4, activation="softmax"))
    my_mlp_mc.compile_model(Losses.CategoricalCrossEntropy, Optimisers.RMSPropOptimizer(lr=0.005, rho=0.5))

    # training the model
    my_mlp_mc.fit(X_mclr, y_mclr, n_iter=100)
    # plotting the decision boundaries
    print_decision(X_mclr,
                   np.argmax(y_mclr, 1),
                   CastMulticlassToBinaryPrediction(my_mlp_mc),
                   "MLP_multiclass_decision_boundary")

def MNIST_ann():
    # Loading the images
    if not os.path.exists("mnist_784_data.pkl"):
        X_mnist, y_mnist = fetch_openml('mnist_784', version=1, return_X_y=True)
        y_mnist = one_hot_encode(y_mnist.astype(int))
        with open('mnist_784_data.pkl', 'wb') as f:
            pickle.dump([X_mnist, y_mnist], f)
    else:
        with open('mnist_784_data.pkl', 'rb') as f:
            X_mnist, y_mnist = pickle.load(f)

    # data splitting
    X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(
        X_mnist, y_mnist, train_size=5000, test_size=10000)

    # Normalization
    scaler = StandardScaler()
    X_train_mnist = scaler.fit_transform(X_train_mnist)
    X_test_mnist = scaler.transform(X_test_mnist)

    my_mlp = Model.NN(X_train_mnist.shape)
    my_mlp.add_layer(Layers.DenseLayer(32, activation="relu"))
    my_mlp.add_layer(Layers.DenseLayer(64, activation="relu"))
    my_mlp.add_layer(Layers.DenseLayer(128, activation="relu"))
    my_mlp.add_layer(Layers.DenseLayer(10, activation="softmax"))
    my_mlp.compile_model(Losses.CategoricalCrossEntropy, Optimisers.RMSPropOptimizer(.0002, rho=0.5))

    my_mlp.fit(X_train_mnist[:1000], y_train_mnist[:1000], n_iter=10, verbose=1)

    plt.plot(range(len(my_mlp.losses)), my_mlp.losses)
    plt.title("Learning curve MNIST")
    plt.savefig("mlp-mnist-loss")

    plt.figure(figsize=(10, 5))
    for i in range(10):
        pred_plot = plt.subplot(2, 5, i + 1)

        pred = np.argmax(my_mlp.predict(X_test_mnist[i:i + 1]), 1)

        pred_plot.imshow(X_test_mnist[i].reshape(28, 28), interpolation='nearest',
                         cmap=plt.cm.gray)
        pred_plot.set_xticks(())
        pred_plot.set_yticks(())
        pred_plot.set_xlabel('Pred: %s, True: %s' % (pred, np.argmax(y_test_mnist[i])))
    plt.suptitle('Classification vector for...')
    plt.savefig('figures/mnist_ann')

def CNN_MNIST():
    # Loading the images
    if not os.path.exists("mnist_784_data.pkl"):
        X_mnist, y_mnist = fetch_openml('mnist_784', version=1, return_X_y=True)
        y_mnist = one_hot_encode(y_mnist.astype(int))
        with open('mnist_784_data.pkl', 'wb') as f:
            pickle.dump([X_mnist, y_mnist], f)
    else:
        with open('mnist_784_data.pkl', 'rb') as f:
            X_mnist, y_mnist = pickle.load(f)

    # data splitting
    X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(
        X_mnist, y_mnist, train_size=5000, test_size=10000)

    # Normalization
    scaler = StandardScaler()
    X_train_mnist = scaler.fit_transform(X_train_mnist)
    X_test_mnist = scaler.transform(X_test_mnist)

    my_cnn = Model.NN((10, 28, 28, 1))
    my_cnn.add_layer(Conv.Conv2DLayer(10, activation="relu"))
    my_cnn.add_layer(Conv.Conv2DLayer(20, activation="relu"))
    my_cnn.add_layer(Conv.FlattenLayer())
    my_cnn.add_layer(Layers.DenseLayer(10, activation="softmax"))
    my_cnn.compile_model(Losses.CategoricalCrossEntropy, Optimisers.RMSPropOptimizer(0.0005, 0.5))

    my_cnn.fit(X_train_mnist[:200].reshape(-1, 28,28,1), y_train_mnist[:200], n_iter=10, verbose=1)
    np.argmax(my_cnn.predict(X_train_mnist[:200].reshape(-1, 28, 28, 1)), 1)

    confusion_matrix(np.argmax(y_train_mnist[:200], 1),
                     np.argmax(my_cnn.predict(X_train_mnist[:200].reshape(-1, 28,28,1)), 1))

    weights = my_cnn.layers[0].W
    plt.figure(figsize=(10, 5))
    for i in range(10):
        l1_plot = plt.subplot(2, 5, i + 1)
        l1_plot.imshow(np.squeeze(weights[i]), interpolation='nearest',
                       cmap=plt.cm.gray)
        l1_plot.set_xticks(())
        l1_plot.set_yticks(())
    plt.suptitle("CNN first layers weights")
    plt.savefig("figures/mnist_cnn_loss")

if __name__ == "__main__":

    perceptron_training()

    #activation_functions()

    ann_training()

    multi_class_ann()

    MNIST_ann()