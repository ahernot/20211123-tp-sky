import random as rd
import numpy as np
import pandas as pd

import os

from classification import Classifiers
from scoring import Metrics




DIRPATH = 'src'



import_path = os.path.join(DIRPATH, 'export.npy')
data = np.load (import_path)
vals_nb = data.shape[0]
# print(data)






###### RANDOM TEST
# pos_percent = 0.3
# pos_nb = int(vals_nb * pos_percent)
# neg_nb = vals_nb - pos_nb

# ones = np.ones((pos_nb), dtype=np.int)
# zeros = np.zeros((neg_nb), dtype=np.int)
# pred_rd = np.concatenate((ones, zeros))
# np.random.shuffle (pred_rd)

# m = Metrics(data[:, -1], pred_rd)




#class Data:

#     def __init__ (self, values, labels):
#         self.values = values
#         self.labels = labels
#         pass







########## LDA
# Create TRAIN and TEST sets
def create_sets(data, train_frac=0.05):
    # Create (train, test)
    data_shuffled = data.copy()
    np.random.shuffle(data_shuffled)
    train_nb = int(vals_nb * train_frac)
    return data_shuffled[:train_nb], data_shuffled[train_nb:]

np.random.shuffle(data)
data_train, data_test = create_sets(data)

# Train LDA
lda = Classifiers.LDA (data_train)
# Test LDA
pred_lda = np.array([lda(x) for x in data_test[:, :-1]])

metrics_lda = Metrics(data_test[:, -1], pred_lda)
print(metrics_lda.f_score())








#qda = Classifiers.QDA (data)





"""
random : ypred = random w 30%
lda: p(X|Y=0), p(X|Y=1) gaussienne same stdev, diff avg => regle de discrim lineaire; diff stdev => qda
regression when fonction lineaire entre les deux
"""
