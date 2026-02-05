import numpy as np

arr1_2d = np.array([[0, 1, 2],
                    [3, 4, 5]])

arr2_2d = np.array([[7, 8, 9],
                    [10, 11, 12]])

two_d_arrs = [arr1_2d, arr2_2d]

# np_two_d_arrs = np.empty((2, 2, 3))
np_two_d_arrs = np.array([(2, 2, 3)])

for i in range(np_two_d_arrs.shape[1]):
    np_two_d_arrs[i] = arr1_2d
    np_two_d_arrs[i] = arr2_2d

# np.append(np_two_d_arrs, arr1_2d, axis=1)
# np.append(np_two_d_arrs, arr2_2d, axis=1)


print(np_two_d_arrs)
