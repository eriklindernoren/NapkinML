from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from napkin_ml import LogisticRegression, PCA
from napkin_ml.utils import Plot


def main():
    # Load dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Reduce to two classes
    X = X[y != 0]
    y = y[y != 0]
    y -= 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = np.mean(y_pred == y_test)
    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X, y,
        title="Logistic Regression",
        accuracy=accuracy,
        legend_labels=data.target_names)

if __name__ == "__main__":
    main()
