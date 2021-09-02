import matplotlib.pyplot as plt
import numpy as np


def one_hot_encode(y):
    v = np.zeros((y.size, len(np.unique(y))))
    v[np.arange(y.size), y.astype(int)] = 1
    return v

def plot_coefs(coef, title='Classification vector for...'):
    plt.figure(figsize=(10, 5))
    scale = np.abs(coef).max()
    for i in range(10):
        l1_plot = plt.subplot(2, 5, i + 1)
        l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                       cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
        l1_plot.set_xticks(())
        l1_plot.set_yticks(())
        l1_plot.set_xlabel('Class %i' % i)
    plt.suptitle(title)
    plt.savefig("figures/" + title)

def print_decision(X, y, clf, title="Decision Boundary"):
    plt.figure()
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax = plt.subplot(111)
    ax.contourf(xx, yy, Z, alpha=.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', s=25)
    plt.title(title)
    plt.savefig("figures/" + title)


def plot_mnist_predictions(clf, X, y,  title='Classification vector for...'):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        pred_plot = plt.subplot(2, 5, i + 1)

        pred = clf.predict(X[i:i+1])

        pred_plot.imshow(X[i].reshape(28, 28), interpolation='nearest',
                         cmap=plt.cm.gray)
        pred_plot.set_xticks(())
        pred_plot.set_yticks(())
        pred_plot.set_xlabel('Pred: %s, True: %s' %(int(pred[0]), np.argmax(y[i])))
    plt.suptitle(title)
    plt.savefig("figures/" + title)
