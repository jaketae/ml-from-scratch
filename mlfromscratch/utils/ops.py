import numpy as np

def euclidean_distance(x1, x2):
    for x in [x1, x2]:
        if not isinstance(x, np.ndarray):
            x = np.asaray(x)
    return np.sqrt(np.sum((x1 - x2)**2))

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy