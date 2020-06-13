from collections import Counter

import numpy as np
from mlfromscratch.utils.ops import euclidean_distance


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = np.asarray(X), np.asarray(y)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [euclidean_distance(x, data) for data in self.X]
        k_idx = np.argsort(distances)[: self.k]
        k_neighbor_labels = self.y[k_idx]
        most_common = Counter(k_neighbor_labels).most_common(1)[0][0]
        return most_common
