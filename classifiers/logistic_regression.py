import numpy as np


from sklearn.linear_model import LogisticRegression as LR


def sigmoid (x):
    return 1 / (1 + np.exp(-x))


def h (x, theta):
    return 1 / (1 + np.exp(-1 * theta * x))

# def J (theta):
#     return -1 / m * np.sum( y * np.log(h(x, theta)) + (1 - y) * np.log(1 - h(x, theta)) )

# Todo
class LogisticRegression:

    def __init__ (self):
        self.logistic_regression = LR()

    def fit (self, vals: np.ndarray, labels: np.ndarray, verbose=True):
        self.logistic_regression.fit(data, labels)

    def predict (self, vals):
        self.logistic_regression.predict(vals)
