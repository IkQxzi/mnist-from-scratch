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

    # Flatten pixel values into a 1D arr
    x_train = x_train.reshape(60_000, 784)
    x_test = x_test.reshape(10_000, 784)

    return (x_train, y_train), (x_test, y_test)


def get_mnist_batches(batch_size: int):
    '''Gets MNIST training and test data, and returns the training data in batches.'''


    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    # np.split() returns list of arrs, not 2D arr, np.stack converts to 2D arr
    x_batch_shape = (int(x_train.shape[0]/batch_size), batch_size, 784)
    x_train_batches = np.reshape(x_train, x_batch_shape)

    # Use np.reshape() instead of np.split() to get 2D arr instead of list[arrs]
    y_batch_shape = (int(y_train.shape[0]/batch_size), batch_size, 10)
    y_train_batches = np.reshape(y_train, shape=y_batch_shape)

    return (x_train_batches, y_train_batches), (x_test, y_test)


(x_train_batches, y_train_batches), (x_test, y_test) = get_mnist_batches(32)

print(f'x_train_batches shape: {x_train_batches.shape}')
print(f'y_train_batches shape: {y_train_batches.shape}')
print(f'x_train_batches:\n {x_train_batches[0, 0, :]}')