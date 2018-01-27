from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from napkin_ml import LDA, PCA
from napkin_ml.utils import Plot, train_test_split, load_iris

def main():
    data = load_iris()
    X = data['data']
    y = data['target']

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
        legend_labels=data['target_names'])

if __name__ == "__main__":
    main()
