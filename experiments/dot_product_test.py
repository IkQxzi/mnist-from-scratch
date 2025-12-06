import numpy as np

dlds = np.array([0.5, 0.2])
dsdz = np.array([[0.16, -0.08, -0.08],
                 [-0.12, 0.24, -0.12]])

print(np.dot(dlds, dsdz))