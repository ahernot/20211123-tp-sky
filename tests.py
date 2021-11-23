import random as rd
import numpy as np
import pandas as pd

import os

from scoring import Metrics



DIRPATH = 'src'
# csv_path = os.path.join(DIRPATH, 'export.csv')
# vals = pd.read_csv(csv_path).to_numpy()[1:, 1:]

path = os.path.join(DIRPATH, 'export.npy')
vals = np.load (path)
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
print(m)




class Classifiers:

    def __init__ (self):
        pass

    @classmethod
    def LDA (cls, data):

        # Learn normal distribution from training data
        sigma = np.cov (vals.T)
        # Extract 0 and 1 classes
        mask_0 = (data[:, -1] == 0)
        data_0 = data[mask_0, :-1]
        mask_1 = (data[:, -1] == 1)
        data_1 = data[mask_1, :-1]
        # Calculate averages
        mu0 = np.average(data_0, axis=0)
        mu1 = np.average(data_1, axis=0)

        w = np.dot (
            np.linalg.inv(sigma),
            mu1 - mu0
        )

        c = np.dot (
            w,
            0.5 * (mu1 + mu0)
        )

        '''
        '''
    
        def classifier (x):
            return int(np.dot(w, x) > c)

        return cls()



Classifiers.LDA(vals)

"""
random : ypred = random w 30%
lda: p(X|Y=0), p(X|Y=1) gaussienne same stdev, diff avg => regle de discrim lineaire; diff stdev => qda
regression when fonction lineaire entre les deux
"""
