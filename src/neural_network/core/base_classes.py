from typing import List, Dict, Union, Optional

import numpy as np


class LayerGeneric:
    '''A generic structure which will be used for both core layers and activation functions.'''

    def __init_subclass__(cls) -> None:
        
        # Calls the __init_subclass__ for all parents (good practice)
        # to prevent overriding parent class behaviour
        super().__init_subclass__()

        # Useful logging for child classes (layers):

        if 'fwd' not in cls.__dict__:
            print(f'WARNING: No fwd() method implemented for layer type: {cls.__name__}')
        
        if 'bwd' not in cls.__dict__:
            print(f'WARNING: No bwd() method implemented for layer type: {cls.__name__}')


    def __init__(self) -> None:
        pass


    def fwd(self, inputs: np.ndarray):
        pass

    def bwd(self, dvalues: np.ndarray):
        pass


class NN(LayerGeneric):

    def etc(self):
        pass

    def fwfd(self) -> None:
        print('the')
        pass


layer = NN()
print(dir(layer))

print(layer.__class__.__name__)    


