import numpy as np
from typing import Callable

import time

from functions import find_minima


class Random:

    def __init__ (self, pos_ratio=0.3):
        self.pos_ratio = pos_ratio
    
    def eval_batch (self, vals_nb: int):
        self.pos_nb = int(vals_nb * self.pos_ratio)
        self.neg_nb = vals_nb - self.pos_nb
        
        ones = np.ones((self.pos_nb), dtype=np.int)
        zeros = np.zeros((self.neg_nb), dtype=np.int)
        pred_rd = np.concatenate((ones, zeros))

        np.random.shuffle (pred_rd)
        return pred_rd
