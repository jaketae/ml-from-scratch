import numpy as np

class LinearRegression:

    def __init__(self):
    	pass

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        theta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
        return [theta.dot(x) for x in X]