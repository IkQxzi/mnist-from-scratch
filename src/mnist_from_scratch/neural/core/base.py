import cupy as cp

from mnist_from_scratch.tools.debugging import warning_style
from dataclasses import dataclass





class Network:
    
    def __init__(self, layers: list[object]) -> None:
        '''Initialise a network using the architecture specified.'''
        pass


    def fwd(self, inputs: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        '''Forward inputs through the network.'''

        self.probs: cp.ndarray
        self.conf_scores: cp.ndarray

        return self.probs, self.conf_scores


    def bwd(self, loss_dvals: cp.ndarray, true_classes: cp.ndarray, *args, **kwargs) -> None:
        '''Backpropogate through the network.'''
        pass


class Layer:
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


    def __init__(self, *args, **kwargs) -> None:
        pass

    def fwd(self, inputs: cp.ndarray):
        pass

    def bwd(self, dvalues: cp.ndarray, *args, **kwargs):
        pass


class Activation(Layer):
    '''A generic activation class for all activation functions.'''

    def __init_subclass__(cls) -> None:
        
        # Calls the __init_subclass__ for all parents
        # to prevent overriding parent class behaviour
        super().__init_subclass__()

        # All activation functions require kwargs to know how to save their params
        if not hasattr(cls, 'kwargs'):
            print(f'WARNING: No kwargs attribute implemented for layer type: {cls.__name__}')

        # All activation functions require fwd() & bwd() to work
        if 'fwd' not in cls.__dict__:
            print(f'WARNING: No fwd() method implemented for layer type: {cls.__name__}')

        if 'bwd' not in cls.__dict__:
            print(f'WARNING: No bwd() method implemented for layer type: {cls.__name__}')


    def fwd(self, inputs: cp.ndarray) -> None:
        '''
        Performs forward pass.

        Attributes:
            outputs -- the outputs of the forward pass.
        '''

        self.outputs: cp.ndarray
        pass


    def bwd(self, chain_dvals: cp.ndarray) -> None: # type: ignore
        '''
        Performs forward pass using _fwd_func().

        Attributes:
            input_dvals -- The derivatives of the output w.r.t inputs.
        '''
        self.input_dvals: cp.ndarray
        pass


# class ActivationOutput(Activation):

#     def bwd(self, chain_dvals: cp.ndarray, true_classes: cp.ndarray) -> None: # pyright: ignore[reportIncompatibleMethodOverride]

#         pass


class Loss():

    def __init_subclass__(cls) -> None:
        
        # Calls the __init_subclass__ for all parents
        # to prevent overriding parent class behaviour
        super().__init_subclass__()

        # All loss functions require forwarding logic for fwd() to work
        if 'calculate' not in cls.__dict__:
            print(f'WARNING: No calculate() method implemented for layer type: {cls.__name__}')

        if 'bwd' not in cls.__dict__:
            print(f'WARNING: No bwd() method implemented for layer type: {cls.__name__}')


    def calculate(self, probs: cp.ndarray, true_classes: cp.ndarray) -> cp.ndarray: # type: ignore
        '''
        Calculates CCE Loss.

        :param probs: The network's prediction after Softmax pass.
        :param true_classes: The correct classes, one-hot-encoded.
        attributes:
        input_dvals: dvals w.r.t inputs.
        '''

        pass


    def bwd(self) -> None:
        '''Calculates the derivative of loss output w.r.t inputs.
        
        Attributes:
            dvalues -- the derivatives of the layer.
        '''

        pass


class Optimiser:

    def __init__(self, network: object, *args, **kwargs):
        '''Save Optimiser hyperparameters.'''
        pass
    

    def update_params(self, *args, **kwargs) -> None:
        '''Update network parameters using calculated gradients/dvals.'''

        pass


# Want to be constant
# Also makes it hashable for .npz
@dataclass(frozen=True)
class Model:

    # Can use __dict__ to get layer attributes
    layers: list[Layer]
    
    # Settings used during training of this model
    # epochs, batch_size, learning_rate, weight_decay, etc.
    hyperparams: dict[str, int | float]

    # name, timestamp, etc.
    metadata: dict[str, str]

    # Model performance on training set
    # avg_acc, avg_loss
    perf: dict[str, int]

