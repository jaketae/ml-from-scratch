import numpy as np
from collections import defaultdict


class NaiveBayes:

    def __init__(self):
        pass

    def fit(self, X, y):
        self.X, self.y = np.asarray(X), np.asarray(y)
        self.classes = np.unique(y)
        self.parameters = defaultdict()
        for c in self.classes:
            X_c = X[y == c]
            mean = X_c.mean(axis=0)
            var = X_c.var(axis=0)
            self.parameters[c] = [mean, var]

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            log_prior = np.log(self._calculate_prior(c))
            log_likelihood = np.sum(np.log(self._calculate_likelihood(c, x)))
            posteriors.append(log_prior + log_likelihood)
        return self.classes[np.argmax(posteriors)]

    def _calculate_prior(self, c):
        return np.mean(self.y == c)

    def _calculate_likelihood(self, c, x):
        mean, var = self.parameters[c]
        return np.exp(- (x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
