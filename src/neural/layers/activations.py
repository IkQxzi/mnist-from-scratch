'''
Created: Sprint 2
Last Modified: Sprint 2

Purpose:
'''

import numpy as np 

from neural.core.base import Activation
from tools.debugging import debug_style


class ReLU(Activation):
    '''
    Clips outputs below 0.
    Formula: max(x, 0)
    '''

    def fwd(self, inputs: np.ndarray) -> None:

        self.inputs = inputs

        self.outputs = np.maximum(inputs, 0)
    
    def bwd(self, chain_dvals: np.ndarray) -> None:
        
        # Derivative is 1 if inputs >= 0, 0 otherwise
        local_input_dvals = np.divide(self.inputs, self.inputs, 
                                      out=np.zeros_like(self.inputs),
                                      where=self.inputs!=0)

        # Chain rule, element-wise as they are the same shape
        self.input_dvals = chain_dvals * local_input_dvals


class Sigmoid(Activation):
    '''
    Squashes inputs to between 0 & 1.
    Formula: 1 / (1 + e^-x)
    '''


    def fwd(self, inputs: np.ndarray) -> None:

        self.inputs = inputs

        exp_vals = np.exp(inputs)
        self.outputs = 1 / (1 + exp_vals)
    

class Tanh(Activation):
    '''
    Squashes inputs to between -1 and 1.
    Formula: (e^x - e^-x) / (e^x + e^-x)
    '''

    def fwd(self, inputs: np.ndarray) -> None:
        
        self.inputs = inputs
        
        pos_exp_vals = np.exp(inputs)
        neg_exp_vals = np.exp(-inputs)

        self.outputs = (pos_exp_vals - neg_exp_vals) / (pos_exp_vals + neg_exp_vals)


class ReLULeaky(Activation):
    '''
    Makes values below 0 small using a leak rate.
    Formula: max(x, leak_rate*x)
    '''


    def __init__(self, leak_rate: float) -> None:
        '''Define leak rate (between 0 & 1) for outputs < 0.'''
    
        if leak_rate < 0:
            raise ValueError('The leak rate must be between 0 & 1!')

        super().__init__()

        self.leak_rate = leak_rate

    def fwd(self, inputs: np.ndarray) -> None: # type: ignore
        
        self.inputs = inputs

        # leak_rate * inputs is only greater in the negative quadrant
        small_outputs = self.leak_rate * inputs
        self.outputs = np.maximum(small_outputs, inputs)


class Softmax(Activation):
    '''
    Convert raw network outputs into a probability distribution.
    '''


    def fwd(self, inputs: np.ndarray) -> None: # type: ignore 
        '''
        Sum exponentiated inputs and divide each exponentiated input by this sum.
        '''        

        self.inputs = inputs

        clip_inputs = np.clip(-1e-6, 1e-6, inputs)
        exp_vals = np.exp(clip_inputs)

        # axis 0 = batch-wise, axis 1 = neuron-wise
        exp_sum = np.sum(exp_vals, axis=1, keepdims=True) 

        self.outputs = exp_vals / exp_sum
    

    def bwd(self, chain_dvals: np.ndarray, true_classes: np.ndarray) -> None: # pyright: ignore[reportIncompatibleMethodOverride]
        '''
        Calculate the derivatives of Softmax w.r.t inputs, using chain rule.

        :param: chain_dvals -- The derivatives of Loss w.r.t Softmax outputs.
        '''

        # si = softmax output for neuron at index i
        # sj = softmax output for neuron at index of true class (1)
        
        # for sj, dval = sj * (1 - sj)
        # for si, dval = -si * sj

        # Might be useful to generate an inverted cor_pred_arr
        # to multiply outputs by, thereby removing all sj from outputs
        # so we can insert our own sj using the separate formula

        # for si
        # multiply each subarr by the corresponding sj, so:
        # self.outputs[0] * rel_pred_list[0], self.outputs[1] * rel_pred_list[1], etc.
        # Can this be done using element-wise product? or dot-product?

        # Get only the outputs at the correct indexes
        # sj (i = j), 0 otherwise
        cor_pred_arr = true_classes * self.outputs
        cor_pred_list = np.sum(cor_pred_arr, axis=1)

        # Remove all sj from the list of outputs
        # so this list comprises of only si (i /= j), and 0 otherwise
        incor_pred_arr = self.outputs - cor_pred_arr

        # Multiply -si by sj by creating an arr with the cor vals on diagonal
        cor_pred_diag = np.diag(cor_pred_list)
        incor_pred_dvals = np.dot(cor_pred_diag, -1 * incor_pred_arr)

        # Dvals at i /= j will be 0, because cor_pred_arr has sj and 0 otherwise
        cor_pred_dvals = cor_pred_arr * (np.ones_like(cor_pred_arr) - cor_pred_arr)

        # Works because at other indexes, values are 0 for both
        local_input_dvals = incor_pred_dvals + cor_pred_dvals
        self.local_input_dvals = local_input_dvals

        # Remove all the zeros from loss_dvals
        # then multiply every sample by the derivative of loss w.r.t correct index input
        cor_loss_dvals = np.sum(chain_dvals, axis=1)
        cor_loss_dvals_diag = np.diag(cor_loss_dvals)
        self.input_dvals = np.dot(cor_loss_dvals_diag, local_input_dvals)


        
        


# def leaky_relu_test():
    
#     print(debug_style('Leaky ReLU Test'))
#     activ_func = ReLULeaky(leak_rate=0.1)

#     # Mix of pos/neg floats
#     test_data = np.random.randn(4, 3)
#     print(f'test data: {test_data}\n')
    
#     # (Should be) floats between 0 & 1
#     activ_func.fwd(self, test_data)
#     print(f'outputs: {activ_func.outputs}\n')

# def relu_test():
#     print(debug_style('ReLU Test'))
#     activ_func = ReLU()

#     # Mix of pos/neg floats
#     test_data = np.random.randn(4, 3)
#     print(f'test data: {test_data}\n')
    
#     # (Should be) floats between 0 & 1
#     activ_func.fwd(self, test_data)
#     print(f'outputs: {activ_func.outputs}\n')



__all__ = ["ReLU", "Sigmoid", "Tanh", "ReLULeaky", "Softmax"]

