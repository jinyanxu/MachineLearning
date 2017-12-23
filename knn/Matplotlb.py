import KNN
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
g,l = KNN.file2maxtrix("datingTestSet2.txt")
g,ranges,minVals = KNN.autoNorm(g)
ax.scatter(g[:,0],g[:,1],15.0*np.array(l),15.0*np.array(l))
plt.show()

