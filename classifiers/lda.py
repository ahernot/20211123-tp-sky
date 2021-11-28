import numpy as np


class LDA:

    def __init__ (self):
        """
        Linear discriminant analysis (LDA)
        Learn normal distribution from training data
        """
        pass

    def fit (self, data: np.ndarray, labels: np.ndarray, verbose=True):
        # Calculate covariance matrix from training data
        self.sigma = np.cov (data.T)
        self.sigma_inv = np.linalg.inv (self.sigma)

        # Extract 0 and 1 classes
        mask_0 = (labels == 0)
        data_0 = data[mask_0]
        mask_1 = (labels == 1)
        data_1 = data[mask_1]

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

    def eval (self, x):
        # Create classifier function (-> {0, 1})
        return int(np.dot(self.w, x) > self.c)

    def eval_seb (self, x):

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

    def eval_batch (self, data: np.ndarray, verbose=False):
        # Fake vectorisation

        pred_list = list()
        data_nb = data.shape[0]

        for i, x in enumerate(data):
            pred_list .append(self.eval_seb(x))

            if verbose:
                print(f'Progress: {round(100*i/data_nb, 6)}%')

        return np.array(pred_list)
