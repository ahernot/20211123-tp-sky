import random as rd
import numpy as np
import pandas as pd

import os

from classification import LDA, QDA, Kernel
from scoring import Metrics




DIRPATH = 'src'



import_path = os.path.join(DIRPATH, 'export.npy')
data = np.load (import_path)
#data[:, :-1] = data[:, :-1]/255.
vals_nb = data.shape[0]






###### RANDOM TEST
# pos_percent = 0.3
# pos_nb = int(vals_nb * pos_percent)
# neg_nb = vals_nb - pos_nb

# ones = np.ones((pos_nb), dtype=np.int)
# zeros = np.zeros((neg_nb), dtype=np.int)
# pred_rd = np.concatenate((ones, zeros))
# np.random.shuffle (pred_rd)

# m = Metrics(data[:, -1], pred_rd)
# print(m.f_score())





########## Create TRAIN and TEST sets
def create_sets(data, train_frac=0.05):
    # Create (train, test)
    data_shuffled = data.copy()
    np.random.shuffle(data_shuffled)
    train_nb = int(vals_nb * train_frac)
    # print(f'training on {train_nb} vals')
    return data_shuffled[:train_nb], data_shuffled[train_nb:]

data_train, data_test = create_sets(data, train_frac=0.3)


########## LDA f1=0.8875528357524567
# Train LDA
# lda = LDA (data_train)
# # Test LDA
# pred_lda = np.array([lda.predict_seb(x) for x in data_test[:, :-1]])
# metrics_lda = Metrics(data_test[:, -1], pred_lda)
# print(metrics_lda.f_score())

# generate masks!!!!!


########## QDA f1=0.9016371262346097
# Train QDA
# qda = QDA (data_train)
# # Test QDA
# pred_qda = np.array([qda.predict(x) for x in data_test[:10000, :-1]])
# metrics_qda = Metrics(data_test[:10000, -1], pred_qda)
# print(metrics_qda.f_score())


########## KERNEL f1=0.8175104862054705
def func (x1: np.ndarray, x2: np.ndarray):
    dist = np.linalg.norm(x1 - x2)
    if dist > 1: return 1 / np.linalg.norm(x1 - x2)
    else: return 1
    # return np.dot(x1, x2.T)

kernel = Kernel (data_train[:100000], func=func)

# pred_kernel_list = list()
# data_test_nb = data_test[:100000].shape[0]
# for i, x in enumerate(data_test[:100000, :-1]):
#     pred_kernel_list .append(kernel.predict(x))
#     print(f'Progress: {round(100*i/data_test_nb, 6)}%')
# pred_kernel = np.array(pred_kernel_list)
pred_kernel = np.array([kernel.predict(x) for x in data_test[:100000, :-1]])
metrics_kernel = Metrics(data_test[:100000, -1], pred_kernel)
print(metrics_kernel)
print(metrics_kernel.f_score())


########## REGRESSION




"""
random : ypred = random w 30%
lda: p(X|Y=0), p(X|Y=1) gaussienne same stdev, diff avg => regle de discrim lineaire; diff stdev => qda
regression when fonction lineaire entre les deux
"""
