# NapkinML

## About
Pocket-sized implementations of machine learning models.

## Table of Contents
- [NapkinML](#napkinml)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [K-Nearest Neighbors](#k-nearest-neighbors)
    + [Linear Regression](#linear-regression)
    + [Linear Discriminant Analysis](#linear-discriminant-analysis)
    + [Logistic Regression](#logistic-regression)
    + [Multilayer Perceptron](#multilayer-perceptron)
    + [Principal Component Analysis](#principal-component-analysis)

## Installation
    $ git clone https://github.com/eriklindernoren/NapkinML
    $ cd NapkinML
    $ sudo python setup.py install

## Implementations
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
    <img src="http://eriklindernoren.se/images/napkin_knn.png" width="640">
</p>
<p align="center">
    Figure: Classification of the Iris dataset with K-Nearest Neighbors.
</p>


### Linear Regression
```python
class LinearRegression():
    def fit(self, X, y):
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]
    def predict(self, X):
        return X.dot(self.w)
```

```
$ python napkin_ml/examples/linear_regression.py
```  
<p align="center">
    <img src="http://eriklindernoren.se/images/napkin_linreg.png" width="640">
</p>
<p align="center">
    Figure: Linear Regression.
</p>

### Linear Discriminant Analysis
```python
class LDA():
    def fit(self, X, y):
        cov_sum = sum([np.cov(X[y == val], rowvar=False) for val in [0, 1]])
        mean_diff = X[y == 0].mean(0) - X[y == 1].mean(0)
        self.w = np.linalg.inv(cov_sum).dot(mean_diff)
    def predict(self, X):
        return 1 * (X.dot(self.w) < 0)
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
            self.w -= lr * (self.predict(X) - y).dot(X)
    def predict(self, X):
        return sigmoid(X.dot(self.w))
```

```
$ python napkin_ml/examples/logistic_regression.py
```  
<p align="center">
    <img src="http://eriklindernoren.se/images/napkin_logreg.png" width="640">
</p>
<p align="center">
    Figure: Classification with Logistic Regression.
</p>

### Multilayer Perceptron
```python
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
```

```
$ python napkin_ml/examples/mlp.py
```  
<p align="center">
    <img src="http://eriklindernoren.se/images/napkin_mlp1.png" width="640">
</p>
<p align="center">
    Figure: Classification of the Iris dataset with a Multilayer Perceptron <br> with one hidden layer.
</p>


### Principal Component Analysis
```python
class PCA():
    def transform(self, X, dim):
        _, e_val, e_vec = np.linalg.svd(X - X.mean(0), full_matrices=True)
        idx = e_val.argsort()[::-1]
        e_vec = e_vec[idx][:dim]
        return X.dot(e_vec.T)
```

```
$ python napkin_ml/examples/pca.py
```  
<p align="center">
    <img src="http://eriklindernoren.se/images/napkin_pca.png" width="640">
</p>
<p align="center">
    Figure: Dimensionality reduction with Principal Component Analysis.
</p>
