import numpy as np
from skimage import exposure
from sklearn.utils import shuffle

NUM_CLASSES = 43

def to_grayscale(X):
    return (0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2])

def normalize(X):
    return (X / 255.).astype(np.float32)

def equalize(X):
    return np.array([exposure.equalize_adapthist(img) for img in X])

def one_hot(y):
    return np.eye(NUM_CLASSES)[y]

def preprocess(X, y=None, training=False):
    X = to_grayscale(X)
    X = normalize(X)
    X = equalize(X)
    X = X[..., np.newaxis]          # shape: (N, 32, 32, 1)

    if y is not None:
        y = one_hot(y)
        if training:
            X, y = shuffle(X, y)

    return X, y