import numpy as np


class QDA_old:

    def __init__ (self):#, data):
        """
        Quadratic discriminant analysis (QDA)
        """
        pass
        
    def fit (self, data: np.ndarray, labels: np.ndarray, verbose=True):
        # Extract 0 and 1 classes
        mask_0 = (labels == 0)
        data_0 = data[mask_0]
        mask_1 = (labels == 1)
        data_1 = data[mask_1]

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

    def eval (self, x):

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

    def eval_batch (self, data: np.ndarray, verbose=False):
        # Fake vectorisation

        pred_list = list()
        data_nb = data.shape[0]

        for i, x in enumerate(data):
            pred_list .append(self.eval(x))

            if verbose:
                print(f'Progress: {round(100*i/data_nb, 6)}%')

        return np.array(pred_list)




def extract_classes_bin (data, labels, class_0 = 0, class_1 = 1):
    mask_0 = (labels == class_0)
    data_0 = data[mask_0]
    mask_1 = (labels == class_1)
    data_1 = data[mask_1]

    return data_0, data_1



from numpy.linalg import inv, det

class QDA:

    def __init__ (self):
        def predit_unit (x: np.ndarray):
            dif_0 = x - self.mu_0
            dif_1 = x - self.mu_1
            likelihood_0 = np.dot( dif_0.T, np.dot( inv(self.sigma_0), dif_0) ) + np.log( det(self.sigma_0) )
            likelihood_1 = np.dot( dif_1.T, np.dot( inv(self.sigma_1), dif_1) ) + np.log( det(self.sigma_1) )
            return int(likelihood_0 > likelihood_1)
        self.predict_vect = np.vectorize(predit_unit)

    def fit (self, X: np.ndarray, y: np.ndarray):
        # Extract 0 and 1 classes
        data_0, data_1 = extract_classes_bin (X, y)

        # Calculate covariance matrices from training data
        self.sigma_0 = np.cov (data_0.T)
        self.sigma_1 = np.cov (data_1.T)

        # Calculate averages from training data
        self.mu_0 = np.average(data_0, axis=0)
        self.mu_1 = np.average(data_1, axis=0)


    def predict (self, X: np.ndarray):
        return self.predict_vect(X)