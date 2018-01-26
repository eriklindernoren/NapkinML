from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from napkin_ml import MLP, PCA
from napkin_ml.utils import Plot

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

    Plot().plot_in_2d(X, y,
        title="Multilayer Perceptron",
        accuracy=accuracy,
        legend_labels=data.target_names)

if __name__ == "__main__":
    main()
