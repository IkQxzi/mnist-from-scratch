
import numpy as np

from neural import NetworkDense, LayerDense


def save_params(network: NetworkDense, path: str, metadata: dict) -> None:
    '''
    Write network parameters (layers, biases) to .npz file.
    The parameters are named layer{n}_{weights/biases} (0-indexed).
    
    :param network: Network to save params from.
    :type network: NetworkDense

    :param path: Name of file to save params to (not incl. extension). 
    :type path: str
    
    :param metadata: Information about how the network was trained.
    :type metadata: dict
    '''

    params = dict()
    
    # Add metadata 
    params = metadata

    # Add weights & biases of network
    for i, layer in enumerate(network.layers):
        if isinstance(layer, LayerDense):
            params[f'layer{i}_weights'] = layer.weights
            params[f'layer{i}_biases'] = layer.biases

    

    print(params.keys())

    np.savez(
            f'{path}.npz', 
            **params
                )
    


    