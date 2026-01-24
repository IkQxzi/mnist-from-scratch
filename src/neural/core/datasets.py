import numpy as np
import cupy as cp
from keras.datasets import mnist

from neural.core.data import create_one_hot

def get_mnist_data() -> tuple:
    '''
    Load MNIST training and test data.

    Returns:
    (x_train, y_train), (x_test, y_test)
    '''

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Convert (np) labels to (cp) one-hot vectors
    y_train = create_one_hot(y_train)
    y_test = create_one_hot(y_test)

    # Convert (np) input data into Cupy float32 arrs
    x_train = cp.asarray(x_train, dtype='float32')
    x_test = cp.asarray(x_test, dtype='float32')

    # Flatten pixel values into a 1D arr
    x_train = x_train.reshape(60_000, 784)
    x_test = x_test.reshape(10_000, 784)

    # Standardise data to (0, 1) (currently (0, 255))
    x_train *= (1/255)
    x_test *= (1/255)

    return (x_train, y_train), (x_test, y_test)


def get_mnist_batches(batch_size: int):
    '''Gets MNIST training and test data, and returns the training data in batches.'''


    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    # Format input data as batches
    x_batch_shape = (int(x_train.shape[0]/batch_size), batch_size, 784)
    x_train_batches = cp.reshape(x_train, x_batch_shape)

    assert cp.all(x_train_batches <= 1), "Pixel values not standardised correctly."

    # Use cp.reshape() instead of cp.split() to get 2D arr instead of list[arrs]
    y_batch_shape = (int(y_train.shape[0]/batch_size), batch_size, 10)
    y_train_batches = cp.reshape(y_train, y_batch_shape)

    return (x_train_batches, y_train_batches), (x_test, y_test)