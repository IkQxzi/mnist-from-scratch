
from neural_network.core import NetworkNeural, LayerDense
from neural_network.tests import NetworkExample

import numpy as np


def scrap_test() -> None:
    n_layers = 2
    n_network_inputs = 4
    
    test_data = np.array([[0.5, 3, 25],
                            [4, 5.3, -3],
                            [4.2, -5, 0.7],
                            [4.3, 0.8, 0.3]])
    print(test_data.shape)
    
    net = NetworkNeural(n_layers, [3, 4], n_network_inputs)
    net.fwd(test_data)


def example_forward_test() -> None:
    # Example input data
    n_layers = 2
    n_network_inputs = 4

    net = NetworkExample(n_layers, n_network_inputs)
    net.forward_test()


def main() -> None:
    # scrap_test()

    loops = 10 
    for x in range(loops):
        example_forward_test()
    print(f'All {loops} loops completed successfully.')


if __name__ == '__main__':
    main()


# main.py