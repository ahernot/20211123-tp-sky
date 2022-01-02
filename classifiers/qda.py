import numpy as np

class QDA_bin:

    def __init__ (self): pass
    
    def fit (self, X: np.ndarray, y: np.ndarray):
        """
        Fit classifier to training data
        :param X: training data
        :param y: training data labels
        """

        # Extract 0 and 1 classes
        X_0 = X[y == 0]
        X_1 = X[y == 1]

        # Covariance matrices
        self.sigma_0 = np.cov (X_0.T)
        self.sigma_1 = np.cov (X_1.T)
        self.sigma_0_inv = np.linalg.inv (self.sigma_0)
        self.sigma_1_inv = np.linalg.inv (self.sigma_1)

        # Calculate averages from training data
        self.mu_0 = np.average(X_0, axis=0)
        self.mu_1 = np.average(X_1, axis=0)

        # Estimate densities
        self.pi_0 = X_0.shape[0] / X.shape[0]
        self.pi_1 = X_1.shape[0] / X.shape[0]

        # Pre-calculate blocks (program optimisation)
        self.lpi_0 = np.log(self.pi_0)  # log(pi0)
        self.lpi_1 = np.log(self.pi_1)  # log(pi1)
        self.mu0_isig0 = np.dot(self.mu_0.T, self.sigma_0_inv)  # mu0.T * sigma0^-1
        self.mu1_isig1 = np.dot(self.mu_1.T, self.sigma_1_inv)  # mu1.T * sigma1^-1


    def predict (self, X: np.ndarray):
        S_0 = np.dot(self.mu0_isig0 , self.mu_0) -2 * np.dot(self.mu0_isig0 , X.T) - 2 * self.lpi_0
        S_1 = np.dot(self.mu1_isig1 , self.mu_1) -2 * np.dot(self.mu1_isig1 , X.T) - 2 * self.lpi_1

        res = (S_0 > S_1).astype(np.int)
        return res
