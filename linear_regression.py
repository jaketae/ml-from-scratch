import numpy as np

class LinearRegression:

    def __init__(self):
    	pass

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        return [theta @ x for x in X]