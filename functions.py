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
