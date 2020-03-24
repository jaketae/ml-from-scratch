import numpy as np
import matplotlib.pyplot as plt
from mlfromscratch.utils.load import cls_data
from mlfromscratch.models.k_means import KMeans


def main(centers=3):
    X, y = cls_data(centers=centers)
    cls = KMeans(k=centers, n_iters=100)
    y_pred = cls.predict(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.set_title("True Cluster")
    ax1.scatter(X[:, 0], X[:, 1], c=y)
    ax2.set_title("Predicted Cluster")
    ax2.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()


if __name__ == '__main__':
    main()
