from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from napkin_ml import MLP, PCA

def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target

    # One-hot
    y = np.zeros((data.target.shape[0], 3))
    y[np.arange(data.target.shape[0]).astype('int'), data.target] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = MLP()
    clf.fit(X_train, y_train, n_epochs=1000, lr=0.01, n_units=16)

    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_pred == y_test)

    y = np.argmax(y, axis=1)

    print ("Accuracy:", accuracy)

    # Reduce dimensions to 2d using pca and plot the results
    X_transformed = PCA().transform(X, 2)

    x1 = X_transformed[:, 0]
    x2 = X_transformed[:, 1]
    class_distr = []

    # Get colors
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    # Plot the different class distributions
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

    # Plot legend
    plt.legend(class_distr, np.unique(y), loc=1)

    # Titles
    perc = 100 * accuracy
    plt.suptitle("Multilayer Perceptron")
    plt.title("Accuracy: %.1f%%" % perc, fontsize=10)

    # Axis labels
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.show()


if __name__ == "__main__":
    main()
