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

### K-Nearest Neighbors
```python
class KNN():
    def predict(self, k, Xt, X, y):
        y_pred = np.empty(len(Xt))
        for i, xt in enumerate(Xt):
            idx = np.argsort([np.linalg.norm(x-xt) for x in X])[:k]
            y_pred[i] = np.bincount([y[i] for i in idx]).argmax()
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
    def transform(self, X, n):
        eval, evec = np.linalg.eig(np.cov(X, rowvar=False))
        idx = eval.argsort()[::-1]
        evec = np.atleast_1d(evec[:, idx])[:, :n]
        return X.dot(evec)
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
