import numpy as np

data = np.array([1, 2, np.nan, 4, 5])
result = np.nanstd(data)

print("Standard deviation ignoring NaN values:", result)
