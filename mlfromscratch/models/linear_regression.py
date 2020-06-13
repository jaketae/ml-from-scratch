import numpy as np


class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        self.weight = np.dot(
            np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.y
        )
        return [np.dot(self.weight, x) for x in X]
