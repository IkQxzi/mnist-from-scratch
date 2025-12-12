import numpy as np


arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

# split into batches of size two
def batch_split(arr: np.ndarray, batch_size: int):
    '''Split a 2D Numpy arr into batches.'''

    arrs = arr[np.arange(start=0, stop=arr.shape[0], step=3), :]
    print(arrs)


batch_split(arr, batch_size=3)