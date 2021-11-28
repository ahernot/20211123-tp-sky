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


def label_freqs (labels: np.ndarray):
    data_len = labels.shape[0]
    freq_0 = np.count_nonzero(labels==0) / data_len
    freq_1 = np.count_nonzero(labels==1) / data_len
    return freq_0, freq_1

def gini_bin (freq_0, freq_1):  # no need for data
    # freq_0, freq_1 = label_freqs(labels=labels)
    return 1 - freq_0**2 - freq_1**2
