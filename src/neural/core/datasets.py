import cupy as cp
from keras.datasets import mnist

from neural.core.data import conv_np_one_hot_cp, batch_split

def get_mnist_data() -> tuple:
    '''
    Load MNIST training and test data.

    Returns:
    (x_train, y_train), (x_test, y_test)
    '''

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Convert (np) labels to (cp) one-hot vectors
    y_train = conv_np_one_hot_cp(y_train)
    y_test = conv_np_one_hot_cp(y_test)

    # Convert (np) input data into Cupy float32 arrs
    x_train = cp.asarray(x_train, dtype='float32')
    x_test = cp.asarray(x_test, dtype='float32')

    # Flatten pixel values into a 1D arr
    x_train.reshape(60_000, 784)
    x_test.reshape(10_000, 784)

    # Standardise data to (0, 1) (currently (0, 255))
    x_train *= (1/255)
    x_test *= (1/255)

    return (x_train, y_train), (x_test, y_test)


# TODO: deprecate
def get_mnist_batches(batch_size: int):
    '''Gets MNIST training and test data, and returns the training data in batches.'''


    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    
    x_train, y_train = batch_split(x_train, batch_size), batch_split(y_train, batch_size)

    return (x_train, y_train), (x_test, y_test)