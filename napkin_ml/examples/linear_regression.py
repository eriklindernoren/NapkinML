from __future__ import print_function
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from napkin_models import LinearRegression

def main():

    X, y = make_regression(n_samples=100, n_features=1, noise=20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    n_samples, n_features = np.shape(X)

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # Print MSE
    print ('MSE: %.4f' % (np.mean(y_test - y_pred)**2))

if __name__ == "__main__":
    main()
