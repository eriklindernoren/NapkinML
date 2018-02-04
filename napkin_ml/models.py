import numpy as np
import random
from scipy.special import expit as sigmoid
from scipy.linalg import svd

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class LinearRegression():
    def fit(self, X, y):
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]
    def predict(self, X):
        return X.dot(self.w)

class KNN():
    def predict(self, k, Xt, X, y):
        y_pred = np.empty(len(Xt))
        for i, xt in enumerate(Xt):
            idx = np.argsort([np.linalg.norm(x-xt) for x in X])[:k]
            y_pred[i] = np.bincount(y[idx]).argmax()
        return y_pred

class PCA():
    def transform(self, X, dim):
        _, S, V = np.linalg.svd(X - X.mean(0), full_matrices=True)
        idx = S.argsort()[::-1]
        V = V[idx][:dim]
        return X.dot(V.T)

class LDA():
    def fit(self, X, y):
        cov_sum = sum([np.cov(X[y == val], rowvar=False) for val in [0, 1]])
        mean_diff = X[y == 0].mean(0) - X[y == 1].mean(0)
        self.w = np.linalg.inv(cov_sum).dot(mean_diff)
    def predict(self, X):
        return 1 * (X.dot(self.w) < 0)

class LogisticRegression():
    def fit(self, X, y, n_iter=4000, lr=0.01):
        self.w = np.random.rand(X.shape[1])
        for _ in range(n_iter):
            self.w -= lr * (self.predict(X) - y).dot(X)
    def predict(self, X):
        return sigmoid(X.dot(self.w))

class MLP():
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

class KMeans():
    def compute_clusters(self, X, centers):
        dist = np.array([np.linalg.norm(X-c, axis=1) for c in centers])
        return np.argmin(dist, axis=0)
    def compute_centers(self, X, clusters):
        return np.array([X[clusters == c,].mean(0) for c in set(clusters)])
    def fit(self, X, k, n_iter=10, random_seed=0):
        clusters = self.compute_clusters(X, np.array(random.sample(list(X), k)))
        for _ in range(n_iter):
            centers = self.compute_centers(X, clusters)
            clusters = self.compute_clusters(X, centers)
        return clusters
