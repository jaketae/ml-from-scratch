import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def clf_data(n_samples=100, n_features=10):
	X, y = datasets.make_classification(n_samples, n_features, random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
	return X_train, X_test, y_train, y_test

def reg_data(n_samples=100, n_features=10):
	X, y, coef = datasets.make_regression(n_samples, n_features, coef=True, random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
	return X_train, X_test, y_train, y_test, coef
