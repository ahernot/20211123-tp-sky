import random as rd
import numpy as np
import pandas as pd

import os

from classification import Random, QDA, Kernel

# from classifiers.random import Random
from classifiers.lda import LDA
# from classifiers.qda import QDA
# from classifiers.kernel import Kernel
from classifiers.nearest_neighbors import NearestNeighbors, NearestNeighborsOptimised, NearestNeighborsOptimised2

from scoring import Metrics




DIRPATH = 'src'


########## Import data
import_path = os.path.join(DIRPATH, 'export.npy')
data = np.load (import_path)
#data[:, :-1] = data[:, :-1]/255.
vals_nb = data.shape[0]








########## Create TRAIN and TEST sets
def create_sets(data, train_frac=0.05):

    # Create (train, test)
    data_shuffled = data.copy()
    np.random.seed(0)
    np.random.shuffle(data_shuffled)
    train_nb = int(vals_nb * train_frac)

    # print(f'training on {train_nb} vals')

    return data_shuffled[:train_nb], data_shuffled[train_nb:]

data_train, data_test = create_sets(data, train_frac=0.3)




###### RANDOM TEST (doesn't work)
RUN_RD = False

if RUN_RD:
    ran = Random (pos_ratio=0.3)
    pred_ran = ran.eval_batch (9e6)

    metrics_ran = Metrics(data, pred_ran)
    print(metrics_ran)






########## LDA f1=0.8875528357524567
RUN_LDA = False
if RUN_LDA:
    data_train_lda = data_train
    data_test_lda  = data_test

    # Train LDA
    lda = LDA()
    lda.fit (data_train_lda)

    # Test LDA
    pred_lda = lda.eval_batch(data_test_lda[:, :-1], verbose=True)
    metrics_lda = Metrics(data_test_lda[:, -1], pred_lda)
    print(metrics_lda.f_score())

# generate masks!!!!!


########## QDA f1=0.9016371262346097
RUN_QDA = False
if RUN_QDA:
    data_train_qda = data_train
    data_test_qda  = data_test[:10000]

    # Train QDA
    qda = QDA()
    qda.fit (data_train_qda)

    # Test QDA
    pred_qda = qda.eval_batch(data_test_qda[:, :-1], verbose=True)
    metrics_qda = Metrics(data_test_qda[:, -1], pred_qda)
    print(metrics_qda.f_score())


########## KERNEL f1=0.8175104862054705
RUN_KERNEL = False
if RUN_KERNEL:
    data_train_kernel = data_train[:100000] # 100000
    data_test_kernel  = data_test[:100000] # 100000

    def func (x1: np.ndarray, x2: np.ndarray):
        dist = np.linalg.norm(x1 - x2)
        if dist > 1: return 1 / np.linalg.norm(x1 - x2)
        else: return 1
        # return np.dot(x1, x2.T)

    # Train kernel
    kernel = Kernel (func=func)
    kernel.fit (data_train_kernel[:, :-1], data_train_kernel[:, -1])

    # Test kernel
    pred_kernel = kernel.eval_batch(data_test_kernel[:, :-1], verbose=True)
    metrics_kernel = Metrics(data_test_kernel[:, -1], pred_kernel)
    print(metrics_kernel)
    print(metrics_kernel.f_score())


########## REGRESSION






########## k-NN f1=0.9025367156208278 (1000train, 1000test)
RUN_KNN = True
if RUN_KNN:
    data_train_knn = data_train[:] # 100000
    data_test_knn = data_test[:] # 100000

    knn = NearestNeighborsOptimised(nb_neighbors=100)
    knn.fit (data_train_knn[:, :-1], data_train[:, -1])
    pred_knn = knn.eval_batch(data_test_knn[:, :-1], verbose=False)
    metrics_knn = Metrics(data_test_knn[:, -1], pred_knn)
    print(metrics_knn)
