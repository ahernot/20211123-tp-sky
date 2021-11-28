import time
import numpy as np
# from functions import label_freqs, gini_bin


#TODO: create parent class for both

def label_freqs (labels: np.ndarray):
    data_len = labels.shape[0]
    if data_len == 0: return 0, 0
    freq_0 = np.count_nonzero(labels==0) / data_len
    freq_1 = np.count_nonzero(labels==1) / data_len
    return freq_0, freq_1

def gini_bin (labels: np.ndarray):  # no need for data
    freq_0, freq_1 = label_freqs(labels=labels)
    return 1 - freq_0**2 - freq_1**2



# might be argmax and not argmin
def choose_split (vals, labels, val_range=range(256)):

    def split_score_axis (split_val, axis, vals, labels):  # on one of the 3 axes
        labels_split_A = labels[vals[:, axis] < split_val]  # can be done with count_nonzero?
        labels_split_B = labels[vals[:, axis] >= split_val]
        data_A_len = labels_split_A.shape[0]
        data_B_len = labels_split_B.shape[0]
        # if data_A_len == 0 or data_B_len == 0: return 1000
        return data_A_len * gini_bin(labels_split_A) + data_B_len * gini_bin(labels_split_B)

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

    @classmethod
    def node (cls, data: np.ndarray, dimension: int, max_depth: int, min_homogeneity: float, parent):
        return cls(data=data, dimension=dimension, max_depth=max_depth, min_homogeneity=min_homogeneity, type_='leaf', depth=parent.depth+1, parent=parent)

    def grow (self):

        # Growth checks 1
        if self.depth == self.max_depth:
            # print('max depth reached')
            return
        if self.homogeneity >= self.min_homogeneity:
            # print('homogeneity target reached')
            return

        # Choose split axis and value
        self.split_axis, self.split_val = choose_split (vals=self.data[:, :-1], labels=self.data[:, -1])        

        # Growth checks 2
        if (self.split_val == 0) or (self.split_val == self.data_nb):
            # print('no split found')
            return
        
        # Update type to node
        if self.type == 'leaf': self.type = 'node'

        # Split data
        data_split_A = self.data[self.data[:, self.split_axis] < self.split_val]
        data_split_B = self.data[self.data[:, self.split_axis] >= self.split_val]

        # Generate children
        self.children = [
            Tree.node(data=data_split_A, dimension=self.dimension, max_depth=self.max_depth, min_homogeneity=self.min_homogeneity, parent=self),
            Tree.node(data=data_split_B, dimension=self.dimension, max_depth=self.max_depth, min_homogeneity=self.min_homogeneity, parent=self)
        ]

        # Grow children (changes type from leaf to node)
        if data_split_A.shape[0] > 1: self.children[0].grow()
        if data_split_B.shape[0] > 1: self.children[1].grow()

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

    # def __repr__ (self):
    #     def print_tree_recur (depth, node_list, print_list):
    #         if node_list == None: return print_list
    #         for node in node_list:
    #             type_ = node.type
    #             indent = '\t' * depth
    #             print_list.append(f'{indent}{type_} – size={node.data.shape[0]}') #, depth={node.depth}')
    #             try:
    #                 print_list += print_tree_recur (depth+1, node.children, print_list)
    #             except: pass
    #         return print_list
    #     return '\n'.join( print_tree_recur (depth=0, node_list=[self], print_list=list()) )

class KDTree:

    def __init__ (self, type_: str, data: np.ndarray, dimension: int, depth: int, parent):
        self.parent = parent
        self.data = data
        self.data_nb = data.shape[0]

        self.dimension = dimension
        self.depth = depth
        self.type = type_

    @classmethod
    def root (cls, data: np.ndarray, dimension: int):
        return cls (type_='root', data=data, dimension=dimension, depth=0, parent=None)

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
        data_split_A = self.data[:self.data_nb//2]
        data_split_B = self.data[self.data_nb//2:]

        # Generate children
        self.children = [
            KDTree.node(data=data_split_A, dimension=self.dimension, parent=self),
            KDTree.node(data=data_split_B, dimension=self.dimension, parent=self)
        ]

        # Grow children
        if self.children[0].type == 'node': self.children[0].grow()
        if self.children[1].type == 'node': self.children[1].grow()


def print_tree (tree):
    def print_tree_recur (depth, node_list):
        if node_list == None: return
        for node in node_list:
            type_ = node.type
            indent = '\t' * depth
            print(f'{indent}{type_} – size={node.data.shape[0]}')
            # print_list.append(f'{indent}{type_} – size={node.data.shape[0]}') #, depth={node.depth}')
            try:
                print_tree_recur (depth+1, node.children)
            except: pass
    print_tree_recur (depth=0, node_list=[tree])


# data = np.array ([
#     [9, 12, 10, 0],
#     [84, 95, 109, 1],
#     [82, 113, 146, 0]
# ])

# tree = Tree (data=data, dimension=3, max_depth=5, min_homogeneity=0.8)
# tree.grow()
# print_tree(tree)
