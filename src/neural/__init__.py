from .core.network import NetworkDense
from .core.data import get_mnist_batches
from . import layers, tests

__all__ = ["NetworkDense", "layers", "tests", "get_mnist_batches"]