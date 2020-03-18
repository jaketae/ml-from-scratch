import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    for x in [x1, x2]:
        if not isinstance(x, np.ndarray):
            x = np.asaray(x)
    return np.sqrt(np.sum((x1 - x2)**2))


class KNN:

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        if not isinstance(y, np.ndarray):
            self.y = np.asarray(y)
        else:
            self.y = y
        self.X = X

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.asarray(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, data) for data in self.X]
        k_idx = np.argsort(distances)[:self.k]
        k_neighbor_labels = y[k_idx]
        most_common = Counter(k_neighbor_labels).most_common(1)[0][0]
        return most_common