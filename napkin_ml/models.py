import numpy as np

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
