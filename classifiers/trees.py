import time
import numpy as np

from functions import label_freqs, gini_bin


def choose_split (vals, labels, freqs, val_range = range(256)):

    def split_score_axis (split_val, axis, vals, labels):  # on one of the 3 axes
        labels_split_A = labels[vals[:, axis] < split_val]  # can be done with count_nonzero?
        labels_split_B = labels[vals[:, axis] >= split_val]
        data_A_len = labels_split_A.shape[0]
        data_B_len = labels_split_B.shape[0]
        return data_A_len * gini_bin(*freqs) + data_B_len * gini_bin(*freqs)

    axis_mins = list()
    for axis in range(3):
        scores = [split_score_axis(split_val=x, axis=axis, vals=vals, labels=labels) for x in val_range]
        axis_mins.append(np.argmin(scores))
    axis = np.argmin(axis_mins)
    split_val = axis_mins[axis]

    return axis, split_val




class Tree:

    def __init__ (self, data: np.ndarray, dimension: int, max_depth: int = -1, min_homogeneity: float = 1., type_: str = 'root', depth: int = 0, parent = None):
        """
        :param data: Node data (values and labels)
        :param dimension: Tree dimension
        :param max_depth: Tree max depth (depth values are in range(0, max_depth+1)), = -1 for unlimited
        :param min_homogeneity: Breakoff homogeneity, = 1. for full tree
        :param type_: Node type
        :param depth: Node depth
        :param parent: Node parent
        """

        # Tree parameters
        self.max_depth       = max_depth
        self.min_homogeneity = min_homogeneity
        self.dimension       = dimension

        # Node data
        self.data    = data
        self.data_nb = data.shape[0]

        # Calculate label frequencies
        self.freqs = label_freqs(self.data[:, -1])
        if self.freqs[0] > self.freqs[1]: self.dominant_label, self.homogeneity = 0, self.freqs[0]
        else:                             self.dominant_label, self.homogeneity = 1, self.freqs[1]

        # Node parameters
        self.type   = type_
        self.depth  = depth
        self.parent = parent

        # Node children (filled if node!=leaf)
        self.children   = None
        self.split_axis = None
        self.split_val  = None


    def __repr__ (self):
        def print_tree_recur (depth, node_list, print_list):
            if node_list == None: return print_list
            for node in node_list:
                type_ = node.type
                indent = '\t' * depth
                print_list.append(f'{indent}{type_} – size={node.data.shape[0]}')
                try:
                    print_list = print_tree_recur (depth+1, node.children, print_list)
                except: pass
            return print_list
        return '\n'.join( print_tree_recur (depth=0, node_list=[self], print_list=list()) )


    def eval (self, x):
        if self.type == 'leaf':
            return self.dominant_label
        else:
            if x[self.split_axis] < self.split_val: return self.children[0].eval(x)
            else:                                   return self.children[1].eval(x)


    def eval_batch (self, data: np.ndarray, verbose=False):
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

    
    def get_depth (self):
        # recursive function
        raise NotImplementedError


class DecisionTree (Tree):

    def __init__ (self, data: np.ndarray, dimension: int, max_depth: int = -1, min_homogeneity: float = 1., type_: str = 'root', depth: int = 0, parent = None):
        """
        :param data: Node data (values and labels)
        :param dimension: Tree dimension
        :param max_depth: Tree max depth (depth values are in range(0, max_depth+1)), = -1 for unlimited
        :param min_homogeneity: Breakoff homogeneity, = 1. for full tree
        :param type_: Node type
        :param depth: Node depth
        :param parent: Node parent
        """

        super(DecisionTree, self).__init__(
            data=data,
            dimension=dimension,
            max_depth=max_depth,
            min_homogeneity=min_homogeneity,
            type_=type_,
            depth=depth,
            parent=parent
        )

    @classmethod
    def node (cls, data: np.ndarray, dimension: int, max_depth: int, min_homogeneity: float, parent):
        return cls(data=data, dimension=dimension, max_depth=max_depth, min_homogeneity=min_homogeneity, type_='leaf', depth=parent.depth+1, parent=parent)

    def grow (self):
        print('growing')

        # Growth checks 1 (tree satisfactory)
        if self.depth == self.max_depth:
            return
        if self.homogeneity >= self.min_homogeneity:
            return

        # Choose split axis and value
        self.split_axis, self.split_val = choose_split (vals=self.data[:, :-1], labels=self.data[:, -1], freqs=self.freqs)        

        # Growth checks 2 (no split found)
        if (self.split_val == 0) or (self.split_val == self.data_nb):
            return
        
        # Update type to node
        if self.type == 'leaf': self.type = 'node'

        # Split data
        data_split_A = self.data[self.data[:, self.split_axis] < self.split_val]
        data_split_B = self.data[self.data[:, self.split_axis] >= self.split_val]

        # Generate children
        self.children = [
            DecisionTree.node(data=data_split_A, dimension=self.dimension, max_depth=self.max_depth, min_homogeneity=self.min_homogeneity, parent=self),
            DecisionTree.node(data=data_split_B, dimension=self.dimension, max_depth=self.max_depth, min_homogeneity=self.min_homogeneity, parent=self)
        ]

        # Grow children (changes type from leaf to node)
        self.children[0].grow()
        self.children[1].grow()


class KDTree (Tree):

    def __init__ (self, data: np.ndarray, dimension: int, type_: str = 'root', depth: int = 0, parent = None):

        super(KDTree, self).__init__(
            data=data,
            dimension=dimension,
            max_depth=-1,
            min_homogeneity=1.,
            type_=type_,
            depth=depth,
            parent=parent
        )

    @classmethod
    def node (cls, data: np.ndarray, dimension: int, max_depth: int, min_homogeneity: float, parent):
        return cls(data=data, dimension=dimension, max_depth=max_depth, min_homogeneity=min_homogeneity, type_='leaf', depth=parent.depth+1, parent=parent)

    def grow (self):

        # Growth checks 1
        if self.depth == self.max_depth:
            return
        if self.homogeneity >= self.min_homogeneity:
            return

        # Choose split axis and value
        self.split_axis = self.depth % self.dimension
        self.split_val = np.median (self.data[:, self.split_axis])

        # Growth checks 2 (no split found)
        if (self.split_val == 0) or (self.split_val == self.data_nb):
            return

        # Update type to node
        if self.type == 'leaf': self.type = 'node'

        # Split data (along median)
        data_split_A = self.data[:self.data_nb//2]
        data_split_B = self.data[self.data_nb//2:]

        # Generate children
        self.children = [
            DecisionTree.node(data=data_split_A, dimension=self.dimension, max_depth=self.max_depth, min_homogeneity=self.min_homogeneity, parent=self),
            DecisionTree.node(data=data_split_B, dimension=self.dimension, max_depth=self.max_depth, min_homogeneity=self.min_homogeneity, parent=self)
        ]

        # Grow children (changes type from leaf to node)
        self.children[0].grow()
        self.children[1].grow()
