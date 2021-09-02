import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import fetch_openml, make_classification, make_blobs, make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from BinaryLogisticRegressor import BinaryLogisticRegression
from MultiClassLogisticRegressor import MultiClassLogisticRegression
from LinearSVM import LinearSVM
from Utils import print_decision, one_hot_encode, plot_mnist_predictions, plot_coefs

def random_data_model():
    # Initialise data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                               random_state=0, n_clusters_per_class=2, class_sep=2.0)
    # Plot the generated data
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', s=25)
    plt.title('Plot of the generated data and their label')
    plt.savefig("figures/" + "random data generation")

    # Train the Binary logistic regression model
    clf = BinaryLogisticRegression(n_iter=100, lr=0.1)
    clf.fit(X, y)

    # Train the Binary logistic regression model using scikit-learn
    clf_sci = LogisticRegression(C=50. / 100 , penalty='l2', tol=0.1, max_iter=100)
    clf_sci.fit(X, y)

    # Print the output
    print_decision(X, y, clf, "boundary-blr")
    print_decision(X, y, clf_sci, "sklearn-boundary-blr")

def blobs_data():
    X_lsvm, y_lsvm = make_blobs(n_samples=50, centers=2)

    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X_lsvm, y_lsvm)

    plt.scatter(X_lsvm[:, 0], X_lsvm[:, 1], c=y_lsvm, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')

    plt.title("SVM and Margin separation")
    plt.savefig("figures/" + "sklearn - SVM and Margin separation")

    my_lsvm = LinearSVM(10000, 0.01, 10)
    my_lsvm.fit(X_lsvm, y_lsvm)
    print_decision(X_lsvm, y_lsvm, my_lsvm, "blob_svm")

    my_blr = BinaryLogisticRegression(1000, 0.1, r_lambda=0)
    my_blr.fit(X_lsvm, y_lsvm)
    print_decision(X_lsvm, y_lsvm, my_blr, "blob_binarylr")

def circles_data():
    X_ksvm, y_ksvm = make_circles(100, factor=.4, noise=.1)

    kernel_svm_clf = svm.SVC(kernel='linear', C=100)
    kernel_svm_clf.fit(X_ksvm, y_ksvm)
    print_decision(X_ksvm, y_ksvm, kernel_svm_clf, "circles-linear-svc")

    kernel_svm_clf = svm.SVC(kernel='poly', C=100)
    kernel_svm_clf.fit(X_ksvm, y_ksvm)
    print_decision(X_ksvm, y_ksvm, kernel_svm_clf, "circles-polynomial-svc")

    kernel_svm_clf = svm.SVC(kernel='rbf', C=100)
    kernel_svm_clf.fit(X_ksvm, y_ksvm)
    print_decision(X_ksvm, y_ksvm, kernel_svm_clf, "circles-gaussian-svc")

def mnist_model():

    train_samples = 5000

    # Loading the images
    if not os.path.exists("mnist_784_data.pkl"):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        y = one_hot_encode(y.astype(int))
        with open('mnist_784_data.pkl', 'wb') as f:
            pickle.dump([X, y], f)
    else:
        with open('mnist_784_data.pkl', 'rb') as f:
            X, y = pickle.load(f)

    # shuffling part
    random_state = check_random_state(0)
    #permutation = random_state.permutation(X.shape[0])
    #X = X[permutation]
    #y = y[permutation]
    #X = X.reshape((X.shape[0], -1))

    # data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_samples, test_size=10000)

    # Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        C=50. / train_samples, penalty='l2', solver='saga', tol=0.1
    )

    clf.fit(X_train, np.argmax(y_train, 1))

    sparsity = np.mean(clf.coef_ == 0) * 100
    score = clf.score(X_test, np.argmax(y_test, 1))

    print("Sparsity with L1 penalty: %.2f%%" % sparsity)
    print("Test score with L1 penalty: %.4f" % score)

    plot_mnist_predictions(clf, X_test, y_test, 'sklearn-mnist-mcr-predictions')
    plot_coefs(clf.coef_.copy(), title='sklearn-mnist-mcr-coefs"')

    # Manually implemented
    my_mclr = MultiClassLogisticRegression(n_iter=100, lr=0.001, regul="l2", r_lambda=0.1)
    my_mclr.fit(X_train, y_train)
    sparsity = np.mean(my_mclr.W == 0) * 100
    score = my_mclr.score(X_test, np.argmax(y_test))

    print("Sparsity: %.2f%%" % sparsity)
    print("Test score: %.4f" % score)

    plot_mnist_predictions(my_mclr, X_test, y_test, 'mnist-mcr-predictions')
    plot_coefs(my_mclr.W.copy(), title='mnist-mcr-coefs"')


if __name__ == "__main__":

    # Train our logistic regressor and sci-kit learns
    random_data_model()

    # Train an MNIST model
    mnist_model()

    # Linear SVM
    blobs_data()

    # SVM Kernels
    circles_data()
