import numpy as np
from keras.datasets import mnist


def create_one_hot(arr: np.ndarray) -> np.ndarray:
    '''Convert a 1D Numpy array to a 2D one-hot encoded vector.'''

    # Ensure arr is 1D
    assert len(arr.shape) == 1, f'Cannot create one hot vector with arr of shape {arr.shape}'

    # Create arr of zeros (len=10)
    arr_one_hot = np.zeros(shape=(arr.shape[0], 10))

    # Place ones at indexes of arr
    arr_one_hot[np.arange(arr_one_hot.shape[0]), arr] = 1
    return arr_one_hot


def get_mnist_data() -> tuple:
    '''
    Load MNIST training and test data.

    Returns:
    (x_train, y_train), (x_test, y_test)
    '''

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Convert labels to one-hot vectors
    y_train = create_one_hot(y_train)
    y_test = create_one_hot(y_test)
    
    return (x_train, y_train), (x_test, y_test)


def get_mnist_batches(batch_size: int):
    
    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    x_train_batches = x_train[]
