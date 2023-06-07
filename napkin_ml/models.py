import numpy as np
import random
from scipy.special import expit as sigmoid
from scipy.spatial.distance import cdist
from scipy.linalg import svd


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class LinearRegression:
    def fit(self, X, y):
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]
    def predict(self, X):
        return X.dot(self.w)


class KNN:
    def predict(self, k, Xt, X, y):
        idx = np.argsort(cdist(Xt, X))[:, :k]
        y_pred = [np.bincount(y[i]).argmax() for i in idx]
        return y_pred


class PCA:
    def transform(self, X, dim):
        _, S, V = np.linalg.svd(X - X.mean(0), full_matrices=True)
        idx = S.argsort()[::-1][:dim]
        return X.dot(V[idx].T)


class LDA:
    def fit(self, X, y):
        cov_sum = sum([np.cov(X[y == val], rowvar=False) for val in [0, 1]])
        mean_diff = X[y == 0].mean(0) - X[y == 1].mean(0)
        self.w = np.linalg.inv(cov_sum).dot(mean_diff)
    def predict(self, X):
        return 1 * (X.dot(self.w) < 0)


class LogisticRegression:
    def fit(self, X, y, n_iter=4000, lr=0.01):
        self.w = np.random.rand(X.shape[1])
        for _ in range(n_iter):
            self.w -= lr * (self.predict(X) - y).dot(X)
    def predict(self, X):
        return sigmoid(X.dot(self.w))


class MLP:
    def fit(self, X, y, n_epochs=4000, lr=0.01, n_units=10):
        self.w = np.random.rand(X.shape[1], n_units)
        self.v = np.random.rand(n_units, y.shape[1])
        for _ in range(n_epochs):
            h_out = sigmoid(X.dot(self.w))
            out = softmax(h_out.dot(self.v))
            self.v -= lr * h_out.T.dot(out - y)
            self.w -= lr * X.T.dot((out - y).dot(self.v.T) * (h_out * (1 - h_out)))
    def predict(self, X):
        return softmax(sigmoid(X.dot(self.w)).dot(self.v))


class KMeans:
    def fit(self, X, k, n_iter=200):
        centers = random.sample(list(X), k)
        for _ in range(n_iter):
            clusters = np.argmin(cdist(X, centers), axis=1)
            centers = np.array([X[clusters == c].mean(0) for c in range(k)])
        return clusters
