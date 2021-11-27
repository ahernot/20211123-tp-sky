import numpy as np

def find_minima (arr: np.ndarray, min_nb: int = 1, max_int = 32767):
    min_nb  = min(arr.shape[0], min_nb)
    arr_c = np.copy(arr)
    minima_ids = list()

    for min_i in range (min_nb):

        # Find argmin of array
        min_id = np.argmin(arr_c, axis=0)
        minima_ids .append(min_id)
        
        # Remove position
        arr_c [min_id] = max_int

    return np.array(minima_ids).astype(np.int)


def export_mask (shape:tuple):

    # mask = mask.reshape(shape)
    pass


# argmin & argmax stolen from https://stackoverflow.com/questions/16945518/finding-the-index-of-the-value-which-is-the-min-or-max-in-python
def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


# def gini (x, data: np.ndarray, labels: np.ndarray, val, axis:int=0):
#     """
#     Gini cost function for binary classification problems
#     """
    
#     # Split along x on specified axis
#     mask = data[:, axis] < x
#     data_split_A, data_split_B = data[mask], data[1-mask]

#     data_split_A_len = data_split_A.shape[0]
#     data_split_B_len = data_split_B.shape[0]
    

#     #data_split_A_0 = data_split_A [data_split_A[:, -1] == 0]
#     data_split_A_1 = data_split_A [data_split_A[:, -1] == 1]
#     #data_split_B_0 = data_split_B [data_split_B[:, -1] == 0]
#     data_split_B_1 = data_split_B [data_split_B[:, -1] == 1]

#     #freq_A_0 = data_split_A_0.shape[0] / data_split_A_len
#     freq_A_1 = data_split_A_1.shape[0] / data_split_A_len
#     #freq_B_0 = data_split_B_0.shape[0] / data_split_B_len
#     freq_B_1 = data_split_B_1.shape[0] / data_split_B_len

#     return 1 - freq_A_1**2 - freq_B_1**2



def gini_bin (labels: np.ndarray):  # no need for data
    data_len = labels.shape[0]
    freq_0 = np.count_nonzero(labels==0) / data_len
    freq_1 = np.count_nonzero(labels==1) / data_len
    return 1 - freq_0**2 - freq_1**2
