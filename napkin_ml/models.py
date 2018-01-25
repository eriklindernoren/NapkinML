import numpy as np
from scipy.special import expit as sigmoid

class LinearRegression():
    def fit(self, X, y):
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    def predict(self, X):
        return X.dot(self.w)

class KNN():
    def predict(self, k, Xt, X, y):
        y_pred = np.empty(len(Xt))
        for i, xt in enumerate(Xt):
            idx = np.argsort([np.linalg.norm(x-xt) for x in X])[:k]
            y_pred[i] = np.bincount([y[i] for i in idx]).argmax()
        return y_pred

class PCA():
    def transform(self, X, n):
        eval, evec = np.linalg.eig(np.cov(X, rowvar=False))
        idx = eval.argsort()[::-1]
        evec = np.atleast_1d(evec[:, idx])[:, :n]
        return X.dot(evec)

class LDA():
    def fit(self, X, y):
        cov_tot = sum([np.cov(X[y == c], rowvar=False) for c in [0, 1]])
        mean_diff = X[y == 0].mean(0) - X[y == 1].mean(0)
        self.w = np.linalg.inv(cov_tot).dot(mean_diff)
    def predict(self, X):
        return [1 * (x.dot(self.w) < 0) for x in X]

class LogisticRegression():
    def fit(self, X, y, n_epochs=4000, lr=0.01):
        self.w = np.random.rand(X.shape[1])
        for i in range(n_epochs):
            self.w -= lr * -(y - sigmoid(X.dot(self.w))).dot(X)
    def predict(self, X):
        return np.round(sigmoid(X.dot(self.w))).astype(int)
