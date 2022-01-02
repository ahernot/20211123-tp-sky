import numpy as np


class LDA_old:

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





class LDA_bin:

    def __init__ (self): pass
    
    def fit (self, X: np.ndarray, y: np.ndarray):
        """
        Fit classifier to training data
        :param X: training data
        :param y: training data labels
        """

        # Covariance matrix
        self.sigma = np.cov (X.T)
        self.sigma_inv = np.linalg.inv (self.sigma)

        # Extract 0 and 1 classes
        X_0 = X[y == 0]
        X_1 = X[y == 1]

        # Calculate averages from training data
        self.mu_0 = np.average(X_0, axis=0)
        self.mu_1 = np.average(X_1, axis=0)

        # Estimate densities
        self.pi_0 = X_0.shape[0] / X.shape[0]
        self.pi_1 = X_1.shape[0] / X.shape[0]


        # Pre-calculate blocks (program optimisation)
        self.lpi_0 = np.log(self.pi_0)  # log(pi0)
        self.lpi_1 = np.log(self.pi_1)  # log(pi1)
        self.mu0_isig = np.dot(self.mu_0.T, self.sigma_inv)  # mu0.T * sigma^-1
        self.mu1_isig = np.dot(self.mu_1.T, self.sigma_inv)  # mu1.T * sigma^-1


    def predict (self, X: np.ndarray):
        S_0 = np.dot(self.mu0_isig , self.mu_0) -2 * np.dot(self.mu0_isig , X) - 2 * self.lpi_0
        S_1 = np.dot(self.mu1_isig , self.mu_1) -2 * np.dot(self.mu1_isig , X) - 2 * self.lpi_1

        # S_0x = np.dot (self.mu0_isig, self.mu_0 - 2*X) - 2 * self.lpi_0
        # S_1x = np.dot (self.mu1_isig, self.mu_1 - 2*X) - 2 * self.lpi_1

        res = (S_0 > S_1).astype(np.int)
        return res
