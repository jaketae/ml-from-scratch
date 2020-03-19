import numpy as np
from sklearn import datasets
from .ops import train_test_split

RANDOM_STATE = 42

def clf_data(n_samples=100, n_features=10):
	X, y = datasets.make_classification(n_samples, n_features, random_state=RANDOM_STATE)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
	return X_train, X_test, y_train, y_test

def reg_data(n_samples=100, n_features=10):
	X, y, coef = datasets.make_regression(n_samples, n_features, coef=True, random_state=RANDOM_STATE)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
	return X_train, X_test, y_train, y_test, coef

def cls_data(n_samples=100, n_features=2, centers=3):
	X, y = datasets.make_blobs(n_samples, n_features, centers=centers, random_state=RANDOM_STATE)
	return X, y
