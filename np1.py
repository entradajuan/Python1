import numpy as np
import matplotlib.pyplot as plt

np1 = np.asarray([[1,2,3,2],[2,4,3,2]])
print(np1)
print(type(np1))
print(np1.shape)

fig = plt.figure()
plt.imshow(np1, cmap='inferno', origin='lower')
plt.show()

np2 =np1.reshape(2,4,1)
print(np2)
print(np2.shape)
print(np2.dtype.name)
plt.imshow(np2, cmap='inferno', origin='lower')
plt.show()

print(np.full((16,16), 8))
print(np.random.random((16,16)))
plt.imshow(np.random.random((16,16)), cmap='inferno', origin='lower')
plt.show()

list2 = ["AAA-%s"%i for i in np.arange(5) if (i%2==0)]
print(list2)

