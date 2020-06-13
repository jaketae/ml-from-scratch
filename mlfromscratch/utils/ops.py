import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def train_test_split(X, y, test_size, random_state):
    data_num = len(y)
    test_num = int(test_size * data_num)
    np.random.seed(random_state)
    index = np.random.permutation(data_num)
    X_train = X[index[:-test_num]]
    X_test = X[index[-test_num:]]
    y_train = y[index[:-test_num]]
    y_test = y[index[-test_num:]]
    return X_train, X_test, y_train, y_test
