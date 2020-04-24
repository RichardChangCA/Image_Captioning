import numpy as np

features = 50
# print(np.sin(2*np.pi))
# print(np.arange(features, dtype=np.float32))
# print(2*np.pi*(np.arange(features, dtype=np.float32)))
# print(np.sin(2*np.pi*(np.arange(features, dtype=np.float32))/features))


upper_limit = 9
low_limit = 0
precision = 4
population_size = (upper_limit - low_limit) * pow(10, precision)

print(population_size)