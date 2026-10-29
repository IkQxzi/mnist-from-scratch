import urllib.request
from pathlib import Path

import cupy as cp

from mnist_from_scratch.neural.core.data import batch_split, conv_one_hot


def _fetch_mnist_data(
    path: Path = Path("~/.keras/datasets/mnist.npz"),
) -> tuple[tuple[cp.ndarray, cp.ndarray], tuple[cp.ndarray, cp.ndarray]]:
    path = path.expanduser()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, path)

    with cp.load(path) as f:
        return (f["x_train"], f["y_train"]), (f["x_test"], f["y_test"])


def get_clean_mnist_data() -> tuple[
    tuple[cp.ndarray, cp.ndarray], tuple[cp.ndarray, cp.ndarray]
]:
    """Get cleaned MNIST training and test data.

    Returns:
    (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = _fetch_mnist_data()

    # Convert (np) labels to (cp) one-hot vectors
    y_train = conv_one_hot(y_train)
    y_test = conv_one_hot(y_test)

    # Convert (np) input data into Cupy float32 arrs
    x_train = cp.asarray(x_train, dtype="float32")
    x_test = cp.asarray(x_test, dtype="float32")

    # Flatten pixel values into a 1D arr
    x_train = x_train.reshape(60_000, 784)
    x_test = x_test.reshape(10_000, 784)

    # Standardise data to (0, 1) (currently (0, 255))
    x_train *= 1 / 255
    x_test *= 1 / 255

    return (x_train, y_train), (x_test, y_test)


# TODO: deprecate
def get_mnist_batches(batch_size: int) -> tuple:
    """Gets MNIST training and test data, and returns the training data in batches."""
    (x_train, y_train), (x_test, y_test) = get_clean_mnist_data()

    x_train, y_train = (
        batch_split(x_train, batch_size),
        batch_split(y_train, batch_size),
    )

    return (x_train, y_train), (x_test, y_test)
