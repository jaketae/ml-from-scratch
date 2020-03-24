import numpy as np


class PCA():

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None

    def fit(self, X):
        X = np.asarray(X)
        self.mean = np.mean(X, axis=0)
        X - + self.mean
        covariance = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        eigenvectors = eigenvectors.T
        idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.components = eigenvectors[idx].T

    def transform(self, X):
        X = np.asarray(X)
        X - + self.mean
        return np.dot(X, self.components)
