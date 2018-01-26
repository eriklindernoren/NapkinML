# NapkinML

## About
Pocket-sized implementations of machine learning models.

## Table of Contents
- [NapkinML](#napkinml)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Linear Regression](#linear-regression)
    + [Linear Discriminant Analysis](#linear-discriminant-analysis)
    + [Logistic Regression](#logistic-regression)
    + [K-Nearest Neighbors](#k-nearest-neighbors)
    + [Principal Component Analysis](#principal-component-analysis)

## Installation
    $ git clone https://github.com/eriklindernoren/NapkinML
    $ cd NapkinML
    $ sudo python setup.py install

## Implementations
### Linear Regression
```python
class LinearRegression():
    def fit(self, X, y):
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    def predict(self, X):
        return X.dot(self.w)
```

```
$ python napkin_ml/examples/linear_regression.py
```  
<p align="center">
    <img src="http://eriklindernoren.se/images/linreg.png" width="640">
</p>
<p align="center">
    Figure: Linear Regression.
</p>

### Linear Discriminant Analysis
```python
class LDA():
    def fit(self, X, y):
        cov_tot = sum([np.cov(X[y == c], rowvar=False) for c in [0, 1]])
        mean_diff = X[y == 0].mean(0) - X[y == 1].mean(0)
        self.w = np.linalg.inv(cov_tot).dot(mean_diff)
    def predict(self, X):
        return [1 * (x.dot(self.w) < 0) for x in X]
```

```
$ python napkin_ml/examples/lda.py
```

### Logistic Regression
```python
class LogisticRegression():
    def fit(self, X, y, n_iter=4000, lr=0.01):
        self.w = np.random.rand(X.shape[1])
        for _ in range(n_iter):
            self.w -= lr * -(y - sigmoid(X.dot(self.w))).dot(X)
    def predict(self, X):
        return np.rint(sigmoid(X.dot(self.w)))
```

```
$ python napkin_ml/examples/logistic_regression.py
```  
<p align="center">
    <img src="http://eriklindernoren.se/images/logreg.png" width="640">
</p>
<p align="center">
    Figure: Logistic Regression.
</p>

### K-Nearest Neighbors
```python
class KNN():
    def predict(self, k, Xt, X, y):
        y_pred = np.empty(len(Xt))
        for i, xt in enumerate(Xt):
            idx = np.argsort([np.linalg.norm(x-xt) for x in X])[:k]
            y_pred[i] = np.bincount(y[idx]).argmax()
        return y_pred
```

```
$ python napkin_ml/examples/knn.py
```  
<p align="center">
    <img src="http://eriklindernoren.se/images/knn.png" width="640">
</p>
<p align="center">
    Figure: Classification with K-Nearest Neighbors.
</p>

### Principal Component Analysis
```python
class PCA():
    def transform(self, X, dim):
        e_val, e_vec = np.linalg.eig(np.cov(X, rowvar=False))
        idx = e_val.argsort()[::-1]
        e_vec = e_vec[:, idx][:, :dim]
        return X.dot(e_vec)
```

```
$ python napkin_ml/examples/pca.py
```  
<p align="center">
    <img src="http://eriklindernoren.se/images/pca.png" width="640">
</p>
<p align="center">
    Figure: Dimensionality reduction with Principal Component Analysis.
</p>
