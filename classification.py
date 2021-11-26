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




class QDA:

    def __init__ (self, data):
        """
        Quadratic discriminant analysis (QDA)
        """
        pass
        
    def fit (self, data: np.ndarray, labels: np.ndarray, verbose=True):
        # Extract 0 and 1 classes
        mask_0 = (labels == 0)
        data_0 = data[mask_0]
        mask_1 = (labels == 1)
        data_1 = data[mask_1]

        # Calculate covariance matrices from training data
        self.sigma_0 = np.cov (data_0.T)
        self.sigma_1 = np.cov (data_1.T)

        # Calculate averages from training data
        self.mu_0 = np.average(data_0, axis=0)
        self.mu_1 = np.average(data_1, axis=0)


        # Calculate estimated densities
        self.pi_0 = data_0.shape[0]
        self.pi_1 = data_1.shape[0]


        self.a_0 = -0.5 * (np.log(np.linalg.det(self.sigma_0)))
        self.a_1 = -0.5 * (np.log(np.linalg.det(self.sigma_1)))

    def eval (self, x):

        b_0 = -0.5 * np.dot (
            np.dot (
                (x - self.mu_0).T,
                np.linalg.inv(self.sigma_0)
            ),
            x - self.mu_0
        )
        delta_0 = self.a_0 + b_0 + np.log(self.pi_0)

        b_1 = -0.5 * np.dot (
            np.dot (
                (x - self.mu_1).T,
                np.linalg.inv(self.sigma_1)
            ),
            x - self.mu_1
        )
        delta_1 = self.a_1 + b_1 + np.log(self.pi_1)

        return int(delta_1 > delta_0)

    def eval_batch (self, data: np.ndarray, activation=np.average, verbose=False):
        # Fake vectorisation

        pred_list = list()
        data_nb = data.shape[0]

        for i, x in enumerate(data):
            pred_list .append(self.eval(x))

            if verbose:
                print(f'Progress: {round(100*i/data_nb, 6)}%')

        return np.array(pred_list)




class Kernel:

    def __init__ (self, func: Callable):
        self.func = func

    def fit (self, data: np.ndarray, labels: np.ndarray, verbose=True):
        self.data = data

        # Extract 0 and 1 classes
        mask_0 = (labels == 0)
        self.data_0 = data[mask_0]
        mask_1 = (labels == 1)
        self.data_1 = data[mask_1]

    def eval (self, x):
        x_arr = np.ones(self.data.shape[0])
        x_arr = np.column_stack((x_arr * x[0], x_arr * x[1], x_arr * x[2]))
        a_0 = np.sum (self.func (x_arr, self.data))

        x_arr_0 = np.ones(self.data_0.shape[0])
        x_arr_0 = np.column_stack((x_arr_0 * x[0], x_arr_0 * x[1], x_arr_0 * x[2]))
        x_arr_1 = np.ones(self.data_1.shape[0])
        x_arr_1 = np.column_stack((x_arr_1 * x[0], x_arr_1 * x[1], x_arr_1 * x[2]))

        s_0 = np.sum (self.func(x_arr_0, self.data_0))
        s_1 = np.sum (self.func(x_arr_1, self.data_1))

        p_0 = s_0 / a_0
        p_1 = s_1 / a_0

        if p_1 >= p_0: return 1
        else: return 0

    def eval_batch (self, data: np.ndarray, activation=np.average, verbose=False):
        # Fake vectorisation

        pred_list = list()
        data_nb = data.shape[0]

        for i, x in enumerate(data):
            pred_list .append(self.eval(x))

            if verbose:
                print(f'Progress: {round(100*i/data_nb, 6)}%')

        return np.array(pred_list)



