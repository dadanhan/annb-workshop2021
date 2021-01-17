#Python script to show the results from 'trainsupervised_nn_fbm.py'. Plot of Hurst exponents from test labels (ground truth) and the estimation from neural networks

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

htest = np.array(pd.read_csv('H_testvalues_n20.csv',header=None),dtype='float')
hnn = np.array(pd.read_csv('H_NNestimated_n20.csv',header=None),dtype='float')

htest = htest[:,0]
hnn = hnn[:,0]

#Gaussian Kernel Density Estimation
minv = 0.
maxv = 1.
X, Y = np.mgrid[minv:maxv:200j, minv:maxv:200j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([htest, hnn])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

plt.figure()
plt.subplot(1,2,1)
plt.plot(htest,hnn,'.')
plt.plot([0,1],[0,1],'r-',linewidth=0.4)
plt.xlabel('H simulated')
plt.ylabel('H estimated')
plt.subplot(1,2,2)
plt.imshow(np.rot90(Z),extent=[minv, maxv, minv, maxv])
plt.xlim([minv, maxv])
plt.ylim([minv, maxv])
plt.xlabel('H simulated')
plt.ylabel('H estimated')
plt.tight_layout()
plt.show()