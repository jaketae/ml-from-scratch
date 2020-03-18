import numpy as np

def cross_entropy(y_true, y_pred):
    n_samples = len(y_true)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    total_entropy = - np.sum(
        np.multiply(y_true, np.log(y_pred)) + 
        np.multiply((1 - y_true), np.log(1 - y_pred)))
    return total_entropy / n_samples

class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.alpha = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for i in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(z)

            if verbose:
                cost = cross_entropy(y, y_pred)
                print("Iteration {0}, Cost: {1}".format(i, cost))
                
            dw = np.dot(X.T, (y_predicted - y)) / n_samples
            db = np.sum(y_predicted - y) / n_samples

            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(z)
        vector_int = np.vectorize(int)
        y_predicted_cls = vector_int(y_predicted > 0.5)
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))