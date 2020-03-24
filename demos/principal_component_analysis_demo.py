import numpy as np
import matplotlib.pyplot as plt
from mlfromscratch.utils.load import cls_data
from mlfromscratch.models.principal_component_analysis import PCA


def main(dims=2):
    X, y = cls_data(n_features=4, centers=dims)
    cls = PCA(dims)
    cls.fit(X)
    X_transformed = cls.transform(X)

    plt.scatter(X_transformed[:, 0],
                X_transformed[:, 1],
                c=y)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()


if __name__ == '__main__':
    main()
