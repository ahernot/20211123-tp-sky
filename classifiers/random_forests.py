import sys
sys.setrecursionlimit(10**6)

import numpy as np


# SEB SAID: bool tree



class RandomForest:
    '''
    Optimised with trees
    '''

    def __init__ (self, nb_neighbors: int = 1):
        self.nb_neighbors = nb_neighbors

    def fit (self, data: np.ndarray, labels: np.ndarray, verbose=True):
        pass

    def eval (self, x, activation=np.average):
        pass

    def eval_batch (self, data: np.ndarray, activation=np.average, verbose=False):
        pass
