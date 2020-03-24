import numpy as np
from mlfromscratch.utils.load import reg_data
from mlfromscratch.models.linear_regression import LinearRegression


def main():
    X_train, X_test, y_train, y_test, coef = reg_data()
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("True Coefficient: {0}\nPredicted Coefficient: {1}".format(
        coef, reg.weight))


if __name__ == '__main__':
    main()
