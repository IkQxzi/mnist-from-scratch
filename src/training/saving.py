
import json

import cupy as cp

from neural.core.network import NetworkDense
from neural.layers import LayerDense


def save_model_params(network: NetworkDense, path: str, training_metadata: dict) -> None:
    '''
    Write network parameters (layers, biases) to .npz file.\n
    The parameters are named layer{n}_{weights/biases} (0-indexed).
    
    :param network: Network to save model_params from.
    :type network: NetworkDense

    :param path: Name of file to save model_params to (incl. extension). 
    :type path: str
    
    :param training_metadata: Information about how the network was trained (epochs, learning_rate, etc.).
    :type training_metadata: dict

    container/.npz file:
        name: str
        w_key/b_key: cp.ndarray
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
    model_params = dict()

    # Generate layer metadata
    for i, layer in enumerate(network.layers):

        metadata['layers'].append({
            'cls_name': layer.__class__.__name__,
            'kwargs': layer.kwargs,
            'keys': {} # Will have w_key, b_key
        })

        # Save param keys for dense layers
        if isinstance(layer, LayerDense):

            # Create unique weight, bias keys for each layer
            w_key = f'l{i}_w'
            b_key = f'l{i}_b'

            # Add weights & biases to model_params using unique key names
            model_params[w_key] = layer.weights
            model_params[b_key] = layer.biases

            keys = {'w_key': w_key, 'b_key': b_key}

            # Add weight & bias keys to layer metadata
            metadata['layers'][i]['keys'] = keys
        
    # Keep training metadata separate from layer metadata
    metadata['training'] = training_metadata

    print(metadata)
    print(json.dumps(metadata))

    # Write data to singular .npz file
    cp.savez(
            file=path,
            metadata=json.dumps(metadata), # Must be a kwarg
            **model_params, # Has to be passed in **kwargs format to be written as expected
                )
    


    