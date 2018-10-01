import matplotlib.pyplot as plt
import numpy as np
data = []
with open("measurements.txt",'r') as f:
    for line in f:
        data.append(float(line.split(' ')[1]))
data = np.asarray(data,dtype=np.float32)
noncuda, cuda = np.split(data,2)
epochs = range(10)
#plt.plot(epochs,cuda,label='CuDNN Enabled')
plt.plot(epochs,noncuda/cuda,label='relation')
plt.legend()
plt.show()