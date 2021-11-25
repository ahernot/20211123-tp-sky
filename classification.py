import numpy as np
from typing import Callable

import time

from functions import find_minima

class LDA:

    def __init__ (self):
        """
        Linear discriminant analysis (LDA)
        Learn normal distribution from training data
        """
        pass

    def fit (self, data: np.ndarray, labels: np.ndarray, verbose=True):
        # Calculate covariance matrix from training data
        self.sigma = np.cov (data.T)
        self.sigma_inv = np.linalg.inv (self.sigma)

        # Extract 0 and 1 classes
        mask_0 = (labels == 0)
        data_0 = data[mask_0]
        mask_1 = (labels == 1)
        data_1 = data[mask_1]

        # Calculate averages from training data
        self.mu_0 = np.average(data_0, axis=0)
        self.mu_1 = np.average(data_1, axis=0)

        # Calculate estimated densities
        self.pi_0 = data_0.shape[0]
        self.pi_1 = data_1.shape[0]

        # LDA condition
        self.w = np.dot ( self.sigma_inv, self.mu_1 - self.mu_0 )
        self.c = np.dot ( self.w, 0.5 * (self.mu_1 + self.mu_0) )


        self.k_01 = np.dot (
            np.dot (
                self.mu_0.T,
                self.sigma_inv
            ),
            self.mu_0
        )
        self.k_11 = np.dot (
            np.dot (
                self.mu_1.T,
                self.sigma_inv
            ),
            self.mu_1
        )

    def eval (self, x):
        # Create classifier function (-> {0, 1})
        return int(np.dot(self.w, x) > self.c)

    def eval_seb (self, x):

        k_00 = -2 * np.dot (
            np.dot (
                self.mu_0.T,
                self.sigma_inv
            ),
            x
        )

        k_10 = -2 * np.dot (
            np.dot (
                self.mu_1.T,
                self.sigma_inv
            ),
            x
        )
        
        k_0 = k_00 + self.k_01 - 2 * np.log (self.pi_0)
        k_1 = k_10 + self.k_11 - 2 * np.log (self.pi_1)

        # Calculate argmin
        if k_0 > k_1: return 1
        else: return 0

    def eval_batch (self, data: np.ndarray, activation=np.average, verbose=False):
        # Fake vectorisation

        pred_list = list()
        data_nb = data.shape[0]

        for i, x in enumerate(data):
            pred_list .append(self.eval_seb(x))

            if verbose:
                print(f'Progress: {round(100*i/data_nb, 6)}%')

        return np.array(pred_list)




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




class LogisticRegression:

    def __init__ (self):
        pass

    def fit (self, data: np.ndarray, labels: np.ndarray, verbose=True):

    def eval (self, x):
        pass

    def eval_batch (self, data: np.ndarray, activation=np.average, verbose=False):
        pass




class NearestNeighbors:
    
    def __init__ (self, nb_neighbors: int = 1):
        self.nb_neighbors = nb_neighbors
        self.trained = False

    def __repr__ (self):
        repr_list = [
            f'{self.nb_neighbors}-NN classifier',
            f'trained: {self.trained}'
            # info on training data
        ]
        return '\n'.join (repr_list)

    def fit (self, data: np.ndarray, labels: np.ndarray):
        self.data = np.copy(data)
        self.labels = labels

    def eval_dumb (self, x):
        #take mean
        x_arr = np.ones(self.data.shape[0])
        x_arr = np.column_stack((x_arr * x[0], x_arr * x[1], x_arr * x[2]))
        dists = np.linalg.norm(self.data - x_arr, axis=1)
        dists_labeled = np.column_stack((dists, self.labels))
        np.sort(dists_labeled)

        label_pred = np.average(dists_labeled[:self.nb_neighbors, -1])
        print(dists_labeled)
        print(label_pred)
        return round(label_pred) # return 0 or 1


    def eval (self, x, activation=np.average):
        
        # Calculate distances array
        x_arr = np.ones(self.data.shape[0])
        x_arr = np.column_stack((x_arr * x[0], x_arr * x[1], x_arr * x[2]))
        dists = np.linalg.norm(self.data - x_arr, axis=1)

        # Find minima ids
        min_indices = find_minima(dists, self.nb_neighbors)
        min_labels = self.labels[min_indices]

        # Create 0-1 label
        label_pred = activation(min_labels)

        return round(label_pred)


    def eval_batch (self, data: np.ndarray, activation=np.average, verbose=False):
        # Fake vectorisation

        pred_list = list()
        data_nb = data.shape[0]

        for i, x in enumerate(data):
            pred_list .append(self.eval(x))

            if verbose:
                print(f'Progress: {round(100*i/data_nb, 6)}%')

        return np.array(pred_list)




class NearestNeighborsOptimised:

    def __init__ (self, nb_neighbors: int = 1):
        self.nb_neighbors = nb_neighbors

        #self.ax_vals = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # f1=0.9025367156208278
        #self.ax_vals = np.array([0, 64, 128, 192, 255])  # f1=0.6547363316690783
        self.ax_vals = np.arange(8, 256, step=16)
        
        self.ax_vals_nb = self.ax_vals.shape[0]

        # faire la liste en fonction des valeurs les plus courantes


    def fit (self, data: np.ndarray, labels: np.ndarray, verbose=True):

        if verbose: time_start = time.time()

        self.data = np.copy(data)
        self.labels = labels

        # Create cube with neighbors
        self.neighbors_arr = np.empty((self.ax_vals_nb, self.ax_vals_nb, self.ax_vals_nb, self.nb_neighbors), dtype=np.int)

        for x_id in range(self.ax_vals_nb):
            for y_id in range(self.ax_vals_nb):
                for z_id in range(self.ax_vals_nb):
                    x_arr = np.zeros((data.shape[0], 3)) + np.array([self.ax_vals[x_id], self.ax_vals[y_id], self.ax_vals[z_id]])
                    self.neighbors_arr [x_id, y_id, z_id] = find_minima(np.linalg.norm(data - x_arr, axis=1), self.nb_neighbors)

        if verbose:
            time_stop = time.time()
            print(f'Fitting completed in {time_stop - time_start} seconds.')

    def eval (self, x, activation=np.average):

        # Calculate distance from every point in the cube (dimension per dimension here)
        x_closest_ids = list()
        for i in range(3):
            x_arr_i = np.ones(self.ax_vals_nb) * x[i]
            ax_distances = np.abs(self.ax_vals - x_arr_i) # one-dim norm
            x_closest_id = np.argmin(ax_distances)
            x_closest_ids.append(x_closest_id)

        # Get pre-calculated nearest neighbors from cube
        min_indices = self.neighbors_arr[x_closest_ids[0], x_closest_ids[1], x_closest_ids[2]].astype(np.int)

        # Create 0-1 label
        min_labels = self.labels[min_indices]
        label_pred = activation(min_labels)

        return round(label_pred)

    
    def eval_batch (self, data: np.ndarray, activation=np.average, verbose=False):
        # Fake vectorisation

        pred_list = list()
        data_nb = data.shape[0]

        for i, x in enumerate(data):
            pred_list .append(self.eval(x))

            if verbose:
                print(f'Progress: {round(100*i/data_nb, 6)}%')

        return np.array(pred_list)
