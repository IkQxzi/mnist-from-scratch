
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
    
    :param training_metadata: Information about how the network was trained (epochs, l_rate, etc.).
    :type training_metadata: dict

    container/.npz file:
        metadata: dict(layers: list, training: dict), 
        layers: list(cls_name: str, kwargs: dict | None, keys: dict)
            keys: dict(w_key, b_key)
        training: dict(training_metadata)
    '''

    # Training and layer metadata
    metadata = {'layers': [],
                'training': {}
                }
    
    # Weights, biases, etc.
    params = dict()

    # Generate layer metadata
    for i, layer in enumerate(network.layers):

        metadata['layers'].append({
            'cls_name': layer.__class__.__name__,
            'kwargs': layer.kwargs
        })

        # Save param keys for dense layers
        if isinstance(layer, LayerDense):

            # Create unique weight, bias keys for each layer
            w_key = f'l{i}_w'
            b_key = f'l{i}_b'

            # Add weights & biases to params using unique key names
            params[w_key] = layer.weights
            params[b_key] = layer.biases

            keys = {'w_key': w_key, 'b_key': b_key}

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
    


    