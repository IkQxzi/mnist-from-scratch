from neural_network.core import NetworkNeural, LayerDense
import numpy as np


def main() -> None:
    
    n_layers = 3
    n_layer_neurons = [4, 3, 3]
    n_network_inputs = 3
    
    test_data = np.array([[0.5, 3, 25],
                            [4, 5.3, -3],
                            [4.2, -5, 0.7],
                            [4.3, 0.8, 0.3]])

    network = NetworkNeural(n_layers, n_layer_neurons, n_network_inputs)
    network.fwd(test_data)


if __name__ == '__main__':
    main()



# main.py