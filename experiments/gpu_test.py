import cupy as cp

# 1. Check if CUDA is even available
print(f"CUDA Available: {cp.is_available()}")

# 2. Check how many GPUs are seen
print(f"GPU Count: {cp.cuda.runtime.getDeviceCount()}")

# 3. Check what GPU is currently 'active'
print(f"Current Device: {cp.cuda.Device()}")

# 4. Check where an array actually lives
x = cp.array([1, 2, 3])
print(f"Array is on: {x.device}")