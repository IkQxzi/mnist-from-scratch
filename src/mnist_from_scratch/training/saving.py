# training.saving.py

import json

import cupy as cp

from neural.core.base import Model
from neural.core.network import NetworkDense
from neural.layers import LayerDense


def save_model_params(model: Model, file_path: str) -> None:
    '''Save Model object to dir: .npz file for layer objects, .json for hyperparams, metadata, performance'''

    config_dict = {}

    for category, val in model.__dict__.items():

        # Hyperparams, metadata, etc.
        if type(val) == dict:
            
            config_dict |= {category:val}

        if category == 'layers':

            cp.savez(rf"{file_path}/layers.npz")
        
        else:
            raise NotImplementedError(f"Cannot save parameter of type: {type(val)}")
    
    with open(rf"{file_path}/config.json") as f:

        json.dump(config_dict, f)

    


    