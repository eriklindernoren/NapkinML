import progressbar
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import pandas as pd

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def min_max_scaler(X):
    """Min max scale the dataset X

    Arguments:
        X {list or array}

    Returns:
        Returns a numpy array
    """
    arr = np.array(X)
    return np.divide([x - arr.min() for x in arr],
        (arr.max() - arr.min()))


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def load_regression():
    data = pd.read_csv('napkin_ml/data/temp_linkoping.txt', sep="\t")

    time = np.atleast_2d(data["time"].as_matrix()).T
    temp = data["temp"].as_matrix()

    return {'data': time, 'target': temp}

def load_iris():
    df = pd.read_csv('napkin_ml/data/iris.csv', index_col=False)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    labels = [
        'Iris-setosa',
        'Iris-versicolor',
        'Iris-virginica'
    ]

    for i, label in enumerate(labels):
        y[y == label] = i

    return {'data': X, 'target': y.astype('float'), 'target_names': labels}

def load_digits():
    df = pd.read_csv('napkin_ml/data/digits.csv', index_col=False)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return {'data': X, 'target': y.astype('float'), 'target_names': y}

class Plot():
    def __init__(self):
        self.cmap = plt.get_cmap('viridis')

    def _transform(self, X, dim):
        e_val, e_vec = np.linalg.eig(np.cov(X, rowvar=False))
        idx = e_val.argsort()[::-1]
        e_vec = e_vec[:, idx][:, :dim]
        return X.dot(e_vec)

    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y=None, title=None, accuracy=None, legend_labels=None):
        X_transformed = self._transform(X, dim=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        class_distr = []

        y = np.array(y).astype(int)

        colors = [self.cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

        # Plot the different class distributions
        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y == l]
            _x2 = x2[y == l]
            _y = y[y == l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

        # Plot legend
        if not legend_labels is None:
            plt.legend(class_distr, legend_labels, loc=1)

        # Plot title
        if title:
            if accuracy:
                perc = 100 * accuracy
                plt.suptitle(title)
                plt.title("Accuracy: %.1f%%" % perc, fontsize=10)
            else:
                plt.title(title)

        # Axis labels
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        plt.show()
