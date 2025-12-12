import numpy as np

# Each 'image' has 6 px values
# 4 images total
arr = np.array([[[1, 2, 3], [4, 5, 6]], 
                [[6, 7, 8], [9, 10, 11]],
                [[12, 13, 14], [15, 16, 17]], 
                [[18, 19, 20], [21, 22, 23]]])

# split into batches of 2

def batch_split(arr: np.ndarray, batch_size: int) -> np.ndarray:
    '''Split a large set of 2D arrs (3D) into batches of 1D arrs (2D).'''

    print(f'arr shape before: {arr.shape}')
    arr = np.reshape(arr, (2, 2, 6))
    print(f'arr shape after: {arr.shape}\n{arr}')

    return arr


arr2 = batch_split(arr, batch_size=2)
print(arr2[0])