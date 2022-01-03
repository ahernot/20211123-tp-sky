from typing import Callable
import numpy as np

class Kernel:

    def __init__ (self, func: Callable):
        self.func = func

    def fit (self, data: np.ndarray, labels: np.ndarray, verbose=True):
        self.data = data

        # Extract 0 and 1 classes
        mask_0 = (labels == 0)
        self.data_0 = data[mask_0]
        mask_1 = (labels == 1)
        self.data_1 = data[mask_1]

    def eval (self, x):
        x_arr = np.ones(self.data.shape[0])
        x_arr = np.column_stack((x_arr * x[0], x_arr * x[1], x_arr * x[2]))
        a_0 = np.sum (self.func (x_arr, self.data))

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

    def eval_batch (self, data: np.ndarray, verbose=False):
        # Fake vectorisation

        pred_list = list()
        data_nb = data.shape[0]

        for i, x in enumerate(data):
            pred_list .append(self.eval(x))

            if verbose:
                print(f'Progress: {round(100*i/data_nb, 6)}%')

        return np.array(pred_list)





class Kernel_bin:

    def __init__ (self, func: Callable):
        self.__func = func  # kernel func to ponderate the distances (centered on 0)
    
    def fit (self, X: np.ndarray, y: np.ndarray):
        """
        Fit classifier to training data
        :param X: training data
        :param y: training data labels
        """

        self.X_train = X
        self.N_train = X.shape[0]

        # Extract 0 and 1 classes
        self.X_0 = X[y == 0]
        self.X_1 = X[y == 1]

        self.N0_train = self.X_0.shape[0]
        self.N1_train = self.X_1.shape[1]


        # X_0.shape[0]
        # X_1.shape[0]

    def predict (self, X: np.ndarray):


        N_test = X.shape[0]
        N_train
        


        np.tile(X, ())
        np.tile(X, n)

        self.__func (self.X_0)
        pass



