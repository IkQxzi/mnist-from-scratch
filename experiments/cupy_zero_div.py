import cupy as cp

# arr = cp.array([[1, 2, 3], 
#                 [4, 5, 0], 
#                 [0, 8, 9]], 
#                 dtype='float32')

# arr2 = cp.divide(arr, arr, 
#                  out=cp.zeros_like(arr),
#                  where=arr!=0,
#                  dtype='float32')


arr = cp.array([[1, 2, 3], 
                [4, 5, 0], 
                [0, 8, 9]], 
                dtype='float32')

arr2 = (arr != 0).astype('float32')

print(arr, '\n', arr2)