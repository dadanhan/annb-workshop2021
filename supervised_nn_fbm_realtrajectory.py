import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data
d = pd.read_csv('rab5traj.csv')
uid = np.unique(d['pid'])
data = d[d['pid']==11]
# data = d[d['pid']==15]
# data = d[d['pid']==24]
# data = d[d['pid']==33]
x = np.array(data['x'])
y = np.array(data['y'])
t = np.array(data['t'])    
#calculate displacements
disps = np.sqrt( np.power(x-x[0],2) + np.power(y-y[0],2) )
#load neural network model
import tensorflow as tf
model = tf.keras.models.load_model("model3dense_n20.h5")
#loop through the data points and estimate the hurst exponent in every window of 20 points
h = []
ht = []
for i in range(10,len(disps)-11):
    inx = disps[(i-10):(i+11)]
    #apply differencing and normalization on the data
    inx = np.array([(inx[1:]-inx[0:-1])/(np.amax(inx)-np.amin(inx))])
    test = model.predict(inx)
    h.append(test[0][0])
    ht.append(t[i])

#plot displacements
plt.figure()
plt.subplot(211)
plt.plot(t,disps)
plt.ylabel('Disp.')
plt.xlim([t[0],t[-1]])
plt.subplot(212)
plt.plot(ht,h)
plt.plot([t[0],t[-1]],[0.5,0.5],'r-',lw=0.4)
plt.ylabel('H')
plt.xlabel('t')
plt.xlim(t[0],t[-1])
plt.ylim(0,1)
plt.tight_layout()
plt.show()