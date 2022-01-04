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
        self.X0_train = X[y == 0]
        self.X1_train = X[y == 1]

        self.N0_train = self.X0_train.shape[0]
        self.N1_train = self.X1_train.shape[0]

    def predict (self, X: np.ndarray):
        N_test = X.shape[0]

        # randomly select n train vectors
        N_train_sel = 10
        index_0 = np.random.choice(self.N0_train, N_train_sel, replace=False)
        index_1 = np.random.choice(self.N1_train, N_train_sel, replace=False)

        X0_train_sel = self.X0_train[index_0]
        X1_train_sel = self.X1_train[index_1]
        X_train_sel = np.concatenate((X0_train_sel, X1_train_sel), axis=0)

        # Tile X_train
        X_train_tiled = np.tile (X_train_sel, (N_test, 1, 1))  # useless?
        X0_train_tiled = np.tile (X0_train_sel, (N_test, 1, 1))
        X1_train_tiled = np.tile (X1_train_sel, (N_test, 1, 1))

        # Tile X_test
        X_test_tiled = np.transpose( np.tile (X, (2 * N_train_sel, 1, 1)), axes=(1, 0, 2))  # useless?
        X_test_tiled_0 = np.transpose( np.tile (X, (N_train_sel, 1, 1)), axes=(1, 0, 2))  # Replicate as many times as there are train samples, and transpose (flip on its side)
        X_test_tiled_1 = np.transpose( np.tile (X, (N_train_sel, 1, 1)), axes=(1, 0, 2))

        # Calculate element-wise distances (compress RGB information)
        dist_all = np.linalg.norm(X_train_tiled  - X_test_tiled  , axis=2)
        dist_0   = np.linalg.norm(X0_train_tiled - X_test_tiled_0, axis=2)
        dist_1   = np.linalg.norm(X1_train_tiled - X_test_tiled_1, axis=2)

        # Apply kernel function weighing
        # HERE

        # Sum across training datapoints
        sum_all = np.sum(dist_all, axis=1)
        sum_0   = np.sum(dist_0  , axis=1)
        sum_1   = np.sum(dist_1  , axis=1)

        p0 = sum_0 / sum_all  # each one is a sum of the distances on all the train vectors of class 0; each one is a normalising factor for the corresponding x_test vector
        p1 = sum_1 / sum_all

        # Estimate labels (using Bayes' estimation rule)
        y_pred = (p0 > p1).astype(int)

        return y_pred


    def predict_in_batches (self, X: np.ndarray, batch_size: int = 100):

        # generate batches of testing data
        # bootstrap for each batch (randomly select training vectors)
        
        #X_batches = 



        pass

    # def predict_looped (self, X: np.ndarray, verbose: bool = False):
    #     # Fake vectorisation

    #     pred_list = list()
    #     N = X.shape[0]

    #     for i, x in enumerate(X):
    #         pred_list .append(self.predict(x))


    #         progress = round(100 * i / N)
    #         if verbose and progress % 1 == 0:
    #             print(f'Progress: {progress}%')

    #     return np.array(pred_list)



# def KERNEL_FUNC (x: np.ndarray):
#     return np.exp(-1 * np.power(x, 2))

# def KERNEL_CAPPED (x: np.ndarray):  # 1/x function, capped in [-1, 1]
#     ret = np.ones(x.shape[0])
#     ret[x <= 1] = np.abs(1 / x)
#     return ret