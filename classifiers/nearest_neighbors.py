import numpy as np
from typing import Callable

from functions import find_minima

import time



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

        time_start = time.time()

        pred_list = list()
        data_nb = data.shape[0]

        for i, x in enumerate(data):
            pred_list .append(self.eval(x))

            if verbose:
                print(f'Progress: {round(100*i/data_nb, 6)}%')

        time_stop = time.time()
        print(f'Evaluating completed in {time_stop - time_start} seconds.')

        return np.array(pred_list)


# TODO
class NearestNeighborsOptimised2:
    '''
    Optimised with k-d trees
    '''

    def __init__ (self, nb_neighbors: int = 1):
        self.nb_neighbors = nb_neighbors

    def fit (self, data: np.ndarray, labels: np.ndarray, verbose=True):
        pass

    def eval (self, x, activation=np.average):
        pass

    def eval_batch (self, data: np.ndarray, activation=np.average, verbose=False):
        pass
