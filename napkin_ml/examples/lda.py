from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from napkin_ml import LDA, PCA
from napkin_ml.utils import Plot

def main():
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Reduce to two classes
    X = X[y != 2]
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = LDA()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = np.mean(y_pred == y_test)

    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X, y,
        title="Linear Discriminant Analysis",
        accuracy=accuracy,
        legend_labels=data.target_names)

if __name__ == "__main__":
    main()
