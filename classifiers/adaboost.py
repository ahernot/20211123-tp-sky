import numpy as np

# class Hedge:

#     def __init__ (self, beta = 0.5, time_steps: int = 150):
#         self. __time_steps = time_steps
#         self.__beta = beta

#     def fit (self, X: np.ndarray, y: np.ndarray):
        
#         # Initialise weights
#         self.__weights = np.ones_like(y)

#         # Run through time steps
#         for t in range (self.__time_steps):
            
#             # Calculate distribution p
#             distrib = self.__weights / np.sum(self.__weights)



#             # Update weights
#             weights_multiplier = np.power(
#                 np.ones_like(self.__weights) * self.__beta,
#                 1 - np.abs(self.__hypotheses[t](X) - y)
#             )
#             self.__weights *= weights_multiplier




def weaklearn (weights, x, y):
    """
    X -> y\in[0,1]
    :return: hypothesis function
    """

    return

class Adaboost:

    # TBC

    def __init__ (self, time_steps = 150):
        self.__time_steps = time_steps
        self.__hypotheses = list()
        self.__betas = list()

    
    def fit (self, X: np.ndarray, y: np.ndarray, weaklearn = None):

        # Initialise weights
        self.__weights = np.ones_like(y)

        # Run through time steps
        for t in range (self.__time_steps):
            
            # Calculate distribution p
            distrib = self.__weights / np.sum(self.__weights)

            # Calculate weak hypothesis function
            self.__hypotheses .append(weaklearn(distrib))

            # Calculate hypothesis error
            error = np.dot( distrib, np.abs(self.__hypotheses[t](X) - y) )
            self.__betas .append(error / (1 - error))

            # Update weights
            weights_multiplier = np.power(
                np.ones_like(self.__weights) * self.__betas[t],
                1 - np.abs(self.__hypotheses[t](X) - y)
            )
            self.__weights *= weights_multiplier


        def decision_func (x):
            a = np.sum(np.multiply(
                -1 * np.log(self.__betas),
                [self.__hypotheses[t](x) for t in range(self.__time_steps)]
            ))
            b = 0.5 * np.sum(-1 * np.log(self.__betas))

            return int(a >= b)

        self.__decision_func = decision_func

    def predict (self, X):
        return self.__decision_func(X)


# what is weaklearn?
# how to "receive loss", "suffer loss" in Hedge?