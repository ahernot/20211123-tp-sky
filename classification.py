import numpy as np
from typing import Callable

class LDA:

    def __init__ (self, data):
        """
        Linear discriminant analysis (LDA)
        Learn normal distribution from training data
        """
        
        # Calculate covariance matrix from training data
        self.sigma = np.cov (data[:, :-1].T)
        self.sigma_inv = np.linalg.inv (self.sigma)

        # Extract 0 and 1 classes
        mask_0 = (data[:, -1] == 0)
        data_0 = data[mask_0, :-1]
        mask_1 = (data[:, -1] == 1)
        data_1 = data[mask_1, :-1]

        # Calculate averages from training data
        self.mu_0 = np.average(data_0, axis=0)
        self.mu_1 = np.average(data_1, axis=0)

        # Calculate estimated densities
        self.pi_0 = data_0.shape[0]
        self.pi_1 = data_1.shape[0]

        # LDA condition
        self.w = np.dot ( self.sigma_inv, self.mu_1 - self.mu_0 )
        self.c = np.dot ( self.w, 0.5 * (self.mu_1 + self.mu_0) )


        self.k_01 = np.dot (
            np.dot (
                self.mu_0.T,
                self.sigma_inv
            ),
            self.mu_0
        )
        self.k_11 = np.dot (
            np.dot (
                self.mu_1.T,
                self.sigma_inv
            ),
            self.mu_1
        )

    def predict (self, x):
        # Create classifier function (-> {0, 1})
        return int(np.dot(self.w, x) > self.c)

    def predict_seb (self, x):

        k_00 = -2 * np.dot (
            np.dot (
                self.mu_0.T,
                self.sigma_inv
            ),
            x
        )

        k_10 = -2 * np.dot (
            np.dot (
                self.mu_1.T,
                self.sigma_inv
            ),
            x
        )
        
        k_0 = k_00 + self.k_01 - 2 * np.log (self.pi_0)
        k_1 = k_10 + self.k_11 - 2 * np.log (self.pi_1)

        # Calculate argmin
        if k_0 > k_1: return 1
        else: return 0





class QDA:

    def __init__ (self, data):
        """
        Quadratic discriminant analysis (QDA)
        """

        
        # Extract 0 and 1 classes
        mask_0 = (data[:, -1] == 0)
        data_0 = data[mask_0, :-1]
        mask_1 = (data[:, -1] == 1)
        data_1 = data[mask_1, :-1]

        # Calculate covariance matrices from training data
        self.sigma_0 = np.cov (data_0.T)
        self.sigma_1 = np.cov (data_1.T)

        # Calculate averages from training data
        self.mu_0 = np.average(data_0, axis=0)
        self.mu_1 = np.average(data_1, axis=0)


        # Calculate estimated densities
        self.pi_0 = data_0.shape[0]
        self.pi_1 = data_1.shape[0]


        self.a_0 = -0.5 * (np.log(np.linalg.det(self.sigma_0)))
        self.a_1 = -0.5 * (np.log(np.linalg.det(self.sigma_1)))


    def predict (self, x):

        b_0 = -0.5 * np.dot (
            np.dot (
                (x - self.mu_0).T,
                np.linalg.inv(self.sigma_0)
            ),
            x - self.mu_0
        )
        delta_0 = self.a_0 + b_0 + np.log(self.pi_0)

        b_1 = -0.5 * np.dot (
            np.dot (
                (x - self.mu_1).T,
                np.linalg.inv(self.sigma_1)
            ),
            x - self.mu_1
        )
        delta_1 = self.a_1 + b_1 + np.log(self.pi_1)

        return int(delta_1 > delta_0)




class Kernel:

    def __init__ (self, train_data, func: Callable):
        self.train_data = train_data
        self.func = func

        # Extract 0 and 1 classes
        mask_0 = (train_data[:, -1] == 0)
        self.data_0 = train_data[mask_0, :-1]
        mask_1 = (train_data[:, -1] == 1)
        self.data_1 = train_data[mask_1, :-1]


    def predict (self, x):
        x_arr = np.ones(self.train_data.shape[0])
        x_arr = np.column_stack((x_arr * x[0], x_arr * x[1], x_arr * x[2]))
        a_0 = np.sum (self.func (x_arr, self.train_data[:, :-1]))

        x_arr_0 = np.ones(self.data_0.shape[0])
        x_arr_0 = np.column_stack((x_arr_0 * x[0], x_arr_0 * x[1], x_arr_0 * x[2]))
        x_arr_1 = np.ones(self.data_1.shape[0])
        x_arr_1 = np.column_stack((x_arr_1 * x[0], x_arr_1 * x[1], x_arr_1 * x[2]))

        s_0 = np.sum (self.func(x_arr_0, self.data_0))
        s_1 = np.sum (self.func(x_arr_1, self.data_1))

        p_0 = s_0 / a_0
        p_1 = s_1 / a_0

        if p_1 >= p_0: return 1
        else: return 0



class LogisticRegression:

    def __init__ (self, train_data: np.ndarray):
        pass

    def predict (self, x):
        pass
