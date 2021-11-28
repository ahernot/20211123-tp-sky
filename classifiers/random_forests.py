import sys
import time
sys.setrecursionlimit(10**6)

import numpy as np

from classifiers.trees import DecisionTree


# SEB SAID: bool tree



class RandomForest:
    '''
    Optimised with trees
    '''

    def __init__ (self, nb_trees: int, sample_size: int = 500, max_depth: int = 1, min_homogeneity: float = 1.):
        # Forest parameters
        self.trees = list()
        self.nb_trees = nb_trees
        self.sample_size = sample_size

        # Tree parameters
        self.max_depth = max_depth
        self.min_homogeneity = min_homogeneity

    def __repr__ (self):
        repr_list = [
            'Random forest object',
            f' Tree nb: {self.nb_trees}',
            f' Sample size: '
        ]
        return x

    def fit (self, vals: np.ndarray, labels: np.ndarray, verbose=True):

        if verbose: time_start = time.time()

        # Bootstrap: random
        for tree_id in range(self.nb_trees):
            data = np.column_stack ((vals, labels))
            data_rdsample = data[np.random.choice( np.arange(0, data.shape[0], dtype=np.int), size=(self.sample_size) )]

            tree = DecisionTree (data=data_rdsample, dimension=3, max_depth=self.max_depth, min_homogeneity=self.min_homogeneity)
            tree.grow()
            self.trees.append(tree)

        if verbose:
            time_stop = time.time()
            print(f'Fitting completed in {time_stop - time_start} seconds.')


    def eval (self, x):
        tree_preds = list()
        for tree_id in range(self.nb_trees):
            tree = self.trees[tree_id]
            tree_preds.append (tree.eval(x))
        return round(np.average(tree_preds))


    def eval_batch (self, vals: np.ndarray, verbose=False):  # Fake vectorisation
        time_start = time.time()

        pred_list = list()
        data_nb = vals.shape[0]

        for i, x in enumerate(vals):
            pred_list .append(self.eval(x))
            if verbose: print(f'Progress: {round(100*i/data_nb, 6)}%')

        time_stop = time.time()
        print(f'Evaluating completed in {time_stop - time_start} seconds.')

        return np.array(pred_list)
