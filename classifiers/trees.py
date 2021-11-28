import numpy as np
from functions import label_freqs, gini_bin

# might be argmax and not argmin
def choose_split (vals, labels, val_range=range(256)):

    def split_score_axis (split_val, axis, vals, labels):  # on one of the 3 axes
        labels_split_A = labels[vals[:, axis] < split_val]  # can be done with count_nonzero?
        labels_split_B = labels[vals[:, axis] >= split_val]
        data_A_len = labels_split_A.shape[0]
        data_B_len = labels_split_B.shape[0]
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
        :param min_homogeneity: Breakoff homogeneity
        :param type_: Node type
        :param depth: Node depth
        :param parent: Node parent
        """

        # Tree parameters
        self.max_depth = max_depth
        self.min_homogeneity = min_homogeneity
        self.dimension = dimension

        # Node data
        self.data = data
        self.data_nb = data.shape[0]

        # Node parameters
        self.type = type_
        self.depth = depth
        self.parent = parent

        self.freqs = label_freqs(self.data[:, -1])  # calculate label frequencies
        if self.freqs[0] > self.freqs[1]: self.dominant_label, self.homogeneity = 0, self.freqs[0]
        else: self.dominant_label, self.homogeneity = 1, self.freqs[1]

        # Node children (filled if node!=leaf)
        self.children = None
        self.split_axis = None
        self.split_val = None

    @classmethod
    def node (cls, data: np.ndarray, dimension: int, max_depth: int, parent):
        return cls(data=data, dimension=dimension, max_depth=max_depth, type_='leaf', depth=parent.depth+1, parent=parent)

    def grow (self):

        # Growth checks 1
        if self.depth == self.max_depth: return  # Max depth reached
        if self.homogeneity >= self.min_homogeneity: return

        # Choose split axis and value
        self.split_axis, self.split_val = choose_split (vals=self.data[:, :-1], labels=self.data[:, -1])

        # Growth checks 2
        # TODO: check if a split is empty!! or if only same value on axis => need to terminate (leaf)
        
        # Update type when grow
        self.type_ = 'node'

        data_split_A = self.data[self.data[:, self.split_axis] < self.split_val]
        data_split_B = self.data[self.data[:, self.split_axis] >= self.split_val]

        # Generate children
        self.children = [
            KDTree.node(data=data_split_A, dimension=self.dimension, parent=self),
            KDTree.node(data=data_split_B, dimension=self.dimension, parent=self)
        ]

        # Grow children (changes type from leaf to node)
        if data_split_A.shape[0] > 1: self.children[0].grow()
        if data_split_B.shape[0] > 1: self.children[1].grow()



# tree = Tree()




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


tree = KDTree.root(data=data, dimension=3)
tree.grow()

def print_tree (tree: KDTree):

    def print_tree_recur (depth, node_list):
        if node_list == None: return

        for node in node_list:
            type_ = node.type
            indent = '\t' * depth
            print(f'{indent}{type_} – size={node.data.shape[0]}') #, depth={node.depth}')
            try:
                print_tree_recur (depth+1, node.children)
            except: pass
    
    print_tree_recur (depth=0, node_list=[tree])


print_tree(tree)