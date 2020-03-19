import numpy as np
from mlfromscratch.utils.load import cls_data
from mlfromscratch.utils.ops import accuracy_score
from mlfromscratch.models.k_means import KMeans


def main():
	X, y = cls_data()
	cls = KMeans(k=3, n_iters=1000)
	y_pred = cls.predict(X)
	print("Accuracy: {}".format(accuracy_score(y, y_pred)))

if __name__ == '__main__':
	main()