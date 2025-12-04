# from .dummy_network import NetworkExample
from .dummy import gen_batch_data, gen_test_one_hot, NetworkDenseExample
from .data import gen_test_data

__all__ = ["gen_batch_data", "gen_test_one_hot", "NetworkDenseExample", "gen_test_data"]