import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

arr = arr.flatten()

arr2 = np.array_split(arr, 2)

print(arr2)

arr2 = np.stack(arr2)

print(arr2)