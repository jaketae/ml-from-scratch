import math
import numpy as np
from utils.activation_functions import Sigmoid

class LogisticRegression:

    def __init__(self, learning_rate, n_iters):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.sigmoid = Sigmoid()

    def _init_weights(self, X):
        self.n_samples, self.n_features = X.shape
        limit = 1 / math.sqrt(self.n_features)
        self.weights = np.random.uniform(-limit, limit, (self.n_features,))
        self.bias = 0

    def fit(self, X, y):
        self._init_weights(X)

        for _ in range(self.n_iters):
            y_predicted = self.sigmoid(np.dot(X, self.weights) + self.bias)
            
            dw = np.dot(X.T, (y_predicted - y)) / self.n_samples
            db = np.sum(y_predicted - y) / self.n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_predicted = self.sigmoid(np.dot(X, self.weights) + self.bias)
        vector_int = np.vectorize(int)
        y_predicted_cls = vector_int(y_predicted > 0.5)
        return y_predicted_cls

