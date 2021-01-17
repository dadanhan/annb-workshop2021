#Python script to train, evaluate and save model to estimate the Hurst exponent from trajectory of fBm.

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from stochastic.processes.continuous import FractionalBrownianMotion

#generate our data for training and testing
nsamples = 10000
ntimes = 20
traindata = np.empty((nsamples,ntimes))
trainlabels = np.empty((nsamples,1))
for i in range(0,nsamples):
    hurst_exp = np.random.uniform(0.,1.)
    fbm = FractionalBrownianMotion(hurst=hurst_exp,t=1,rng=None)
    x = fbm.sample(ntimes)
    #apply differencing and normalization on the data
    dx = (x[1:]-x[0:-1])/(np.amax(x)-np.amin(x))
    traindata[i,:] = dx
    trainlabels[i,:] = hurst_exp
testdata = np.empty((nsamples,ntimes))
testlabels = np.empty((nsamples,1))
for i in range(0,nsamples):
    hurst_exp = np.random.uniform(0.,1.)
    fbm = FractionalBrownianMotion(hurst=hurst_exp,t=1,rng=None)
    x = fbm.sample(ntimes)
    dx = (x[1:]-x[0:-1])/(np.amax(x)-np.amin(x))
    testdata[i,:] = dx
    testlabels[i,:] = hurst_exp
np.savetxt("H_testvalues_n"+str(ntimes)+".csv",testlabels,delimiter=",")

print('training data shape:',traindata.shape,'training labels shape:', trainlabels.shape,'test data shape:',testdata.shape,'test labels shape:',testlabels.shape)

#create the model for a fully-connected network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(ntimes,activation='relu',input_shape=(ntimes,)),
    tf.keras.layers.Dense(ntimes-1,activation='relu'),
    tf.keras.layers.Dense(ntimes-2,activation='relu'),
    tf.keras.layers.Dense(1,activation='relu')
])
#add optimizer, a loss function and metrics#
optimizer = 'adam'
# optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer=optimizer,
              loss='mean_absolute_error',
              metrics=['mean_absolute_error','mean_squared_error']
)
model.summary()
#train the model
EPOCHS = 100
history = model.fit(traindata,trainlabels,epochs=EPOCHS,validation_split=0.6,verbose=1)

#Save model
print("Saving model")
model.save("./model3dense_n"+str(ntimes)+".h5")
del model
model = tf.keras.models.load_model("./model3dense_n"+str(ntimes)+".h5")

#evaluate the model generalizes by using the test data set
loss, mae, mse = model.evaluate(testdata, testlabels, verbose=1)
print("Testing set Mean Abs Error: {:5.2f}".format(mae))
#predict values using data in the testing set
test_predictions = model.predict(testdata)
#save predicted values
np.savetxt("H_NNestimated_n"+str(ntimes)+".csv",test_predictions,delimiter=",")