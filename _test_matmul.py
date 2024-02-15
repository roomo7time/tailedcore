import numpy as np

n_samples = 1624
n_pixels = 784
fea_dim = 1024

# Example arrays
array1 = np.random.rand(
    n_pixels, n_samples, fea_dim
)  # Replace a, b, c with actual dimensions
array2 = np.random.rand(fea_dim, n_pixels)

# Multiplying the arrays
result = np.matmul(array1, array2)

# The shape of result will be (a, b, b)
print(result.shape)
