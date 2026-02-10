import numpy as np
import cupy as cp

# TODO: Remake this so it is not trash
# Also test how it handles non-perfect divisor batch sizes
def batch_split(arr: cp.ndarray, batch_size: int) -> cp.ndarray:
    '''Split an array into batches of size specified.'''

    x_batch_shape = (int(arr.shape[0]/batch_size), batch_size, 784)
    arr_batches = cp.reshape(arr, x_batch_shape)

    return arr_batches


def conv_np_one_hot_cp(arr: np.ndarray) -> cp.ndarray:
    '''Convert a 1D Numpy array to a 2D one-hot encoded Cupy vector.'''

    # Ensure arr is 1D
    assert len(arr.shape) == 1, f'Cannot create one hot vector with arr of shape {arr.shape}'

    # Create arr of zeros (len=10)
    # Needs to stay float32 otherwise will be upcasted to float64 in dot product
    arr_one_hot = cp.zeros(shape=(arr.shape[0], 10), dtype='float32')

    # Place ones at indexes of arr
    arr_one_hot[cp.arange(arr_one_hot.shape[0], dtype='int32'), arr] = 1
    return arr_one_hot

def get_dummy_data() -> tuple[np.ndarray, np.ndarray]:
    '''Generate 3 samples of dummy data for 3 classes.'''

    dummy_input = np.array([
        [ 2.0, -1.0, -1.0],  # Strong feature 0 -> Should be Class 0
        [-1.0,  2.0, -1.0],  # Strong feature 1 -> Should be Class 1
        [-1.0, -1.0,  2.0]   # Strong feature 2 -> Should be Class 2
    ])

    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return dummy_input, y_true