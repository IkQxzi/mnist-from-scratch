
import numpy as np

from neural.core import NetworkDense


def save_params(network: NetworkDense, path: str) -> None:
    '''
    Write 3-layer network parameters (layers, biases) to .npz file, e.g. layer[num]_weights.
    
    :param network: Network to save params from.
    :type network: NetworkDense

    :param path: Name of file to save params to (not incl. extension). 
    :type path: str
    '''

    weights = []
    biases = []
    for dense_layer in network.layers[0::2]:
        
        weights.append(dense_layer.weights)
        biases.append(dense_layer.biases)
    
    np.savez(file=f'{path}.npz', 
                layer1_weights=weights[0], layer1_biases=biases[0],
                layer2_weights=weights[1], layer2_biases=biases[1],
                layer3_weights=weights[2], layer3_biases=biases[2]
                )
    


    