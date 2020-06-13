import numpy as np
from mlfromscratch.models.logistic_regression import LogisticRegression
from mlfromscratch.utils.load import clf_data
from mlfromscratch.utils.ops import accuracy_score


def main():
    X_train, X_test, y_train, y_test = clf_data()
    clf = LogisticRegression(learning_rate=0.001, n_iters=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))


if __name__ == "__main__":
    main()
