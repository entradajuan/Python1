import numpy as np
import matplotlib.pyplot as plt

np1 = np.asarray([[1,2,3,2],[2,4,3,2]])
print(np1)
print(np1.shape)


fig = plt.figure()
plt.imshow(np1, cmap='inferno', origin='lower')
plt.show()

