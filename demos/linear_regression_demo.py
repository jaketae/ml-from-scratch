import numpy as np
from mlfromscratch.utils.load import reg_data
from mlfromscratch.models.linear_regression import LinearRegression

def main():
	X_train, X_test, y_train, y_test, coef = reg_data()
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)
	print("True Coefficient: {0}\nPredicted Coefficient: {1}".format(coef, regressor.weight))

if __name__ == '__main__':
	main()