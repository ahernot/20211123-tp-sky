import random as rd
import numpy as np
import pandas as pd

import os

from scoring import Metrics



DIRPATH = 'src'



import_path = os.path.join(DIRPATH, 'export.npy')
vals = np.load (import_path)
vals_nb = vals.shape[0]
# print(vals)






# Generate random
pos_percent = 0.3
pos_nb = int(vals_nb * pos_percent)
neg_nb = vals_nb - pos_nb

ones = np.ones((pos_nb), dtype=np.int)
zeros = np.zeros((neg_nb), dtype=np.int)
pred_rd = np.concatenate((ones, zeros))
np.random.shuffle (pred_rd)

m = Metrics(vals[:, -1], pred_rd)




class Classifiers:

    @classmethod
    def LDA (cls, data):
        """
        Linear discriminant analysis (LDA)
        Learn normal distribution from training data
        """
        
        # Calculate covariance matrix from training data
        sigma = np.cov (data[:, :-1].T)

        # Extract 0 and 1 classes
        mask_0 = (data[:, -1] == 0)
        data_0 = data[mask_0, :-1]
        mask_1 = (data[:, -1] == 1)
        data_1 = data[mask_1, :-1]

        # Calculate averages from training data
        mu_0 = np.average(data_0, axis=0)
        mu_1 = np.average(data_1, axis=0)

        # LDA condition
        w = np.dot ( np.linalg.inv(sigma), mu_1 - mu_0 )
        c = np.dot ( w, 0.5 * (mu_1 + mu_0) )

        # Create classifier function (-> {0, 1})
        def classifier (x):
            return int(np.dot(w, x) > c)

        return classifier


    @classmethod
    def QDA (cls, data):
        """
        Quadratic discriminant analysis (QDA)
        """

        # Extract 0 and 1 classes
        mask_0 = (data[:, -1] == 0)
        data_0 = data[mask_0, :-1]
        mask_1 = (data[:, -1] == 1)
        data_1 = data[mask_1, :-1]

        # Calculate covariance matrices from training data
        sigma_0 = np.cov (data_0.T)
        sigma_1 = np.cov (data_1.T)

        # Calculate averages from training data
        mu_0 = np.average(data_0, axis=0)
        mu_1 = np.average(data_1, axis=0)

        # Create classifier function (-> {0, 1})
        def classifier (x, threshold=0):

            formula = [
                np.dot(
                    np.dot(
                        (x - mu_0).T,
                        np.linalg.inv(sigma_0)
                    ),
                    (x - mu_0)
                ),
                np.log(np.linalg.det(sigma_0)),
                -1 * np.dot(
                    np.dot(
                        (x - mu_1).T,
                        np.linalg.inv(sigma_1)
                    ),
                    (x - mu_1)
                ),

                -1 * np.log(np.linalg.det(sigma_1)),
            ]

            return (np.sum(formula) > threshold)

        return classifier





lda = Classifiers.LDA (vals)
qda = Classifiers.QDA (vals)

"""
random : ypred = random w 30%
lda: p(X|Y=0), p(X|Y=1) gaussienne same stdev, diff avg => regle de discrim lineaire; diff stdev => qda
regression when fonction lineaire entre les deux
"""
