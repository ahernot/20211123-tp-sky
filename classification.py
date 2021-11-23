import numpy as np

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
