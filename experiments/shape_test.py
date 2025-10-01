import numpy as np

arr = np.array([[3, 5, 3],
               [2, 5, 2]])

arr2 = arr.T

print(arr.shape)
print(arr2.shape)

output = np.dot(arr, arr2)
print(output.shape)

arr3 = np.array([[4, 5, 2]])

print(f'arr3.shape: {arr3.shape}')

