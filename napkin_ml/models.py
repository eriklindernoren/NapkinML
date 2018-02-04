import numpy as np
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
    
    def compute_clusters(self, x, centers):
        self.distances = [np.linalg.norm(X-center, axis=1) for center in centers]        
        self.cluster = np.argmin(self.distances, axis=0)
        return np.array(self.cluster)

    def compute_centers(self, X, clusters):
        self.new_centers = [X[np.where(clusters == cluster),].mean() for cluster in np.unique(clusters)]
        return np.array(self.new_centers)

    def fit(self, X, num_clusters, n_iter=1000, random_seed=0):
        self.initial_centers = X[np.random.choice(range(X.shape[0]), size=num_clusters, replace=False),]
        self.clusters = self.compute_clusters(X, self.initial_centers)

        for iteration in range(n_iter):
            self.centers = self.compute_centers(X, self.clusters)
            self.clusters = self.compute_clusters(X, self.centers)

        return self.clusters
