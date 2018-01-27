from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from napkin_ml import KNN, PCA
from napkin_ml.utils import Plot, load_iris, train_test_split

def main():
    data = load_iris()
    X = data['data']
    y = data['target'].astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = KNN()
    y_pred = clf.predict(5, X_test, X_train, y_train)

    accuracy = np.mean(y_pred == y_test)

    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X, y,
        title="K-Nearest Neighbors",
        accuracy=accuracy,
        legend_labels=data['target_names'])

if __name__ == "__main__":
    main()
