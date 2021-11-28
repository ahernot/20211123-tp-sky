import numpy as np

from classifiers.trees import Tree, KDTree, print_tree




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

tree = Tree(data=data, dimension=3)
tree.grow()
print_tree(tree)
