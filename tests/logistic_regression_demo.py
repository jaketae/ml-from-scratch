import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlfromscratch.models.logistic_regression import LogisticRegression

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

regressor = LogisticRegression(learning_rate=0.001, n_iters=1000)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))