import numpy as np

y_true = np.eye(3, 10)

pred_indices = [0, 3, 2] # Only one correct

print(np.where(y_true[:, pred_indices] == 1))
bool_mask = np.array(y_true[:, pred_indices] == 1)
print(bool_mask)
print(np.sum(bool_mask))


print(y_true)
np.random.shuffle(y_true)
print(y_true)