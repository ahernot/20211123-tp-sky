import sys
sys.setrecursionlimit(10**6)

import numpy as np

# from functions import argmin


def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


# SEB SAID: bool tree

#data_plotted = 
# split
#then gini
def gini (x, data: np.ndarray, labels: np.ndarray, val, axis:int=0):
    """
    Gini cost function for binary classification problems
    """
    
    # Split along x on specified axis
    mask = data[:, axis] < x
    data_split_A, data_split_B = data[mask], data[1-mask]

    data_split_A_len = data_split_A.shape[0]
    data_split_B_len = data_split_B.shape[0]
    

    #data_split_A_0 = data_split_A [data_split_A[:, -1] == 0]
    data_split_A_1 = data_split_A [data_split_A[:, -1] == 1]
    #data_split_B_0 = data_split_B [data_split_B[:, -1] == 0]
    data_split_B_1 = data_split_B [data_split_B[:, -1] == 1]

    #freq_A_0 = data_split_A_0.shape[0] / data_split_A_len
    freq_A_1 = data_split_A_1.shape[0] / data_split_A_len
    #freq_B_0 = data_split_B_0.shape[0] / data_split_B_len
    freq_B_1 = data_split_B_1.shape[0] / data_split_B_len

    return 1 - freq_A_1**2 - freq_B_1**2



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






        

class kdTree:

    def __init__ (self, type_: str, data: np.ndarray, dimension: int, depth: int, parent):
        self.parent = parent
        self.data = data
        self.data_nb = data.shape[0]

        self.dimension = dimension
        self.depth = depth
        self.type = type_

    @classmethod
    def head (cls, data: np.ndarray, dimension: int):
        return cls (type_='head', data=data, dimension=dimension, depth=0, parent=None)

    @classmethod
    def node (cls, data: np.ndarray, dimension: int, parent):
        if data.shape[0] <= 1 or np.all(data[:, (parent.depth+1) % dimension] == data[0, (parent.depth+1) % dimension]):  # terminate growth if not enough data or same data (inseparable)
            return cls.leaf(data, dimension, parent=parent)
        else:
            return cls(type_='node', data=data, dimension=dimension, depth=parent.depth+1, parent=parent)

    @classmethod
    def leaf (cls, data: np.ndarray, dimension: int, parent):
        return cls(type_='leaf', data=data, dimension=dimension, depth=parent.depth+1, parent=parent)
        

    def grow (self):
        axis = self.depth % self.dimension
        self.median = np.median (self.data[:, axis])

        # Split along med on axis
        # mask = self.data[:, axis] <= self.median
        # data_split_A, data_split_B = self.data[mask], self.data[~mask]
        data_split_A = self.data[:self.data_nb//2]
        data_split_B = self.data[self.data_nb//2:]

        self.children = [
            kdTree.node(data=data_split_A, dimension=self.dimension, parent=self),
            kdTree.node(data=data_split_B, dimension=self.dimension, parent=self)
        ]
        if self.children[0].type == 'node': self.children[0].grow()
        if self.children[1].type == 'node': self.children[1].grow()



data = np.array([
    [0, 1, 2],
    [1, 2, 3],
    [0, 3, 4],
    [9, 3, 2],
    [8, 3, 4],
    [8, 5, 4],
    [8, 3, 1],
    [4, 5, 4],
    [0, 3, 1],
    [9, 3, 9],
    [1, 1, 1]
])


tree = kdTree.head(data, dimension=3)
tree.grow()

def print_tree (tree: kdTree):

    def print_tree_recur (depth, node_list):
        for node in node_list:
            type_ = node.type
            indent = '\t' * depth
            print(f'{indent}{type_} – size={node.data.shape[0]}') #, depth={node.depth}')
            try:
                print_tree_recur (depth+1, node.children)
            except: pass
    
    print_tree_recur (depth=0, node_list=[tree])


print_tree(tree)
