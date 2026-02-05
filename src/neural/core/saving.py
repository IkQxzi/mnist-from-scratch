
import json

import cupy as cp

from neural.core.network import NetworkDense
from neural.layers import LayerDense


def save_params(network: NetworkDense, path: str, training_metadata: dict) -> None:
    '''
    Write network parameters (layers, biases) to .npz file.\n
    The parameters are named layer{n}_{weights/biases} (0-indexed).
    
    :param network: Network to save params from.
    :type network: NetworkDense

    :param path: Name of file to save params to (not incl. extension). 
    :type path: str
    
    :param metadata: Information about how the network was trained (epochs, l_rate, etc.).
    :type metadata: dict
    '''

    # Training and layer metadata
    metadata = {'layers': [],
                'training': {}
                }
    
    # Weights, biases, etc.
    params = dict()

    # Generate layer metadata
    for i, layer in enumerate(network.layers):

        metadata['layers'][i] = {
            'type': layer.__class__.__name__
        }

        # For leaky relu, etc.
        if hasattr(layer, 'kwargs'):

            metadata['layers'][i] = {
                'type': layer.__class__.__name__,
                'kwargs': layer.kwargs
            }

        if isinstance(layer, LayerDense):

            # Create keys to weights & biases for dense layers
            w_key = f'layer{i}_weights'
            b_key = f'layer{i}_biases'

            # Add weights & biases using unique key names
            params[f'layer{i}_weights'] = layer.weights
            params[f'layer{i}_biases'] = layer.biases

            keys = {'weight_key': w_key, 'bias_key': b_key}

            # Add weight & bias keys to layer metadata
            metadata['layers'][i]['keys'] = keys
        
    # Keep training metadata separate from layer metadata
    metadata['training'] = training_metadata

    # Write data to singular .npz file
    cp.savez(
            f'{path}.npz', 
            json.dumps(metadata),
            **params # Has to be passed in **kwargs format to be written as expected
                )
    


    