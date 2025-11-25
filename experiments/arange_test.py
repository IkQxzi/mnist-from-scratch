import numpy as np

'''
Generate random numbers between 0 and output_size so that we can use them diagonally:
predicted class example:
[[10 20 50 20]
 [40 20 10 30]]
shape: (2, 4)

one-hot example:
[[0 0 1 0]
 [0 0 0 1]]
shape: (2, 4)

in the form of one_hot_indices:
[2, 3]

pseudocode:

one_hot_indices: list[int] = []
for _ in range(pred_classes.shape(1)) # number of samples
    generate number from 0 to pred_class.shape(2) # second axis of shape
    add number to one_hot_indices

use one_hot_indices to generate a 2d arraylike the example
'''

def gen_one_hot_matrix(one_hot_indices: list[int], pred_classes: np.ndarray): 
    '''
    one_hot_indices: indexes of each 1
    arr_size: size of each subarr (no of neurons in final layer)
    '''

    one_hot_arr = np.zeros_like(pred_classes)

    for i, sub_ind in zip(one_hot_indices, range(len((one_hot_arr)))):
        one_hot_arr[sub_ind, i] = 1

    return one_hot_arr



pred_classes = np.array([[0.25, 0.1, 0.3, 0.35],
                        [0.4, 0.1, 0.2, 0.3]])


one_hot_indices = [2, 3]
arr_size = 4
gen_one_hot_matrix(one_hot_indices, pred_classes)

def one_hot(indices: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(indices), n_classes), dtype=int)
    out[[x for x in range(len(indices))], indices] = 1
    return out

y = np.random.randint(0, 4, size=8)
print(y)
x = one_hot(y, 4)
print(x)
