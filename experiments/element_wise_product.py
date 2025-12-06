import numpy as np

rel_pred_list = np.array([2, 3])

output_arr = np.array([[1, 2, 3],
                      [4, 5, 6]])

rel_pred_list = np.diag(rel_pred_list)
print(rel_pred_list)

rel_pred_list = np.array([[0, 2],
                          [3, 0]])

print(np.dot(rel_pred_list, output_arr))




