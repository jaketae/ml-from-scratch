import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)