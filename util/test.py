from conjQP_multi import conjQP_multi
import numpy as np

m1 = np.array([0, 0, 0, 0.3, 0, 0, 0.5, 0.2])[np.newaxis]
m2 = np.array([0, 0, 0.3, 0, 0, 0, 0.4, 0.3])[np.newaxis]

m = np.concatenate((m1,m2), axis = 0)
print(m)

mc_1 = conjQP_multi(m, 'q')
print("commonality combined:", mc_1)
mc_2 = conjQP_multi(m, 'pl')


print("plausibility combined:", mc_2)