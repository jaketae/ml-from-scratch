import numpy as np
from mlfromscratch.utils.activation import sigmoid

class LogisticRegression():

    def __init__(self, learning_rate, n_iters):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def _init_weights(self, X):
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self._init_weights(self.X)

        for _ in range(self.n_iters):
            y_predicted = sigmoid(np.dot(X, self.weights) + self.bias)
            
            dw = np.dot(X.T, (y_predicted - y)) / self.n_samples
            db = np.sum(y_predicted - y) / self.n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        y_pred_cls = np.vectorize(int)(y_pred > 0.5)
        return y_pred_cls

