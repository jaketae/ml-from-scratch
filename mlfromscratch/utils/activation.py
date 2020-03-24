import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) - 1


def relu(x):
    return np.where(x >= 0, x, 0)
