from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from napkin_ml import KMeans, PCA
from napkin_ml.utils import Plot, load_iris, train_test_split

def main():
    data = load_iris()
    X = data['data']
    y = data['target'].astype('int')

    kmeans = KMeans()
    clusters = kmeans.fit(X, k=3)

    Plot().plot_in_2d(X, clusters, "K-Means")

if __name__ == "__main__":
    main()
