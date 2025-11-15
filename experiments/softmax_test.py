import numpy as np


inputs = np.array([[ 0.01591285,  0.02849737],
 [-0.00405091,  0.01209003],
 [-0.02368261, -0.00164378],
 [-0.02231193, -0.00851939],
 [ 0.02833388,  0.01619963],
 [-0.04055687, -0.03512177],
 [ 0.06375383,  0.03172984],
 [ 0.00939769,  0.01889315],
 [ 0.05082169,  0.03112838]])

print(inputs)


exp_vals = np.exp(inputs)
exp_sum = np.sum(exp_vals, axis=1, keepdims=True)

zero_sum = exp_sum = np.sum(exp_vals, axis=0, keepdims=True)
one_sum = exp_sum = np.sum(exp_vals, axis=1, keepdims=True)

print(zero_sum - one_sum)

outputs = exp_vals / exp_sum

print(outputs)
print(np.sum(outputs, axis=1))

print(np.sum(exp_vals, axis=0))
print(np.sum(exp_vals, axis=1))