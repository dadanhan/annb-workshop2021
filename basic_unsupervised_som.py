#Code for a basic unsupervised neural network (Self-Organized Map) applied on images.
#Daniel Han 13 Jan 2021

#We need the OpenCV package to read in images to Python quickly. This is just personal preference and you can use other packages to read in images.
import cv2
#These are packages we used previously in the basic_supervised.py code.
import numpy as np
import matplotlib.pyplot as plt

#read in image
filename = 'hannah_er.tif'
im = cv2.imread(filename,-1)
#crop and normalize image
im = im[250:350,250:350]
im = im/np.amax(im)
#show image
plt.figure()
plt.imshow(im,cmap=plt.cm.binary)
plt.colorbar()
plt.show()

#First we need to initialize a set of neurons with weights corresponding to the number of inputs we will have.
isize = 3  #neuron input size (isize by isize matrix from pixels)
num_neurons = 2  #the number of neurons in the self-organized map
#initialize the neurons
neuronv = np.array(np.random.uniform(0,1,size=(num_neurons,isize,isize)),dtype='float')
# print(neuronv)

#now we will loop through many times repeating the process of adjusting weights and make our neurons compete to match pixels
nloops = 3000   #number of loops
r,c = im.shape  #dimensions of the image
gap = int(np.floor(isize/2))     #offset needed to make sure we don't sample outside the image
for t in range(0,nloops):
    #set the gain co-efficient
    alpha = 0.5/(t+1)
    #sample a part of the image (a square of dimensions isize by isize)
    rsam = np.random.randint(gap,r-gap)
    csam = np.random.randint(gap,c-gap)
    sample = im[(rsam-gap):(rsam+gap+1),(csam-gap):(csam+gap+1)]
    #find the distance between the sample and the neurons
    distances = np.empty(num_neurons)
    for i in range(0,num_neurons):
        distances[i] = np.sum(np.power(neuronv[i]-sample,2))
    #change the neuron weights according to self-organizing map algorithm
    bmu = neuronv[np.argmin(distances)]
    for i in range(0,num_neurons):
        neuronv[i] += alpha*(sample-neuronv[i])*np.exp(-np.sum(np.power(bmu-neuronv[i],2)))

#After training the neurons competitively, match each pixel in the image to its closest neuron and obtain a SOM image that has classified each pixel
somim = np.empty((r-2*gap,c-2*gap))
for i in range(gap,r-gap):
    for j in range(gap,c-gap):
        sample = im[(i-gap):(i+gap+1),(j-gap):(j+gap+1)]
        distances = np.empty(num_neurons)
        for k in range(0,num_neurons):
            distances[k] = np.sum(np.power(neuronv[k]-sample,2))
        somim[i-gap,j-gap] = np.argmin(distances)

#plot the result
plt.figure()
plt.subplot(1,2,1)
plt.imshow(im,cmap=plt.cm.binary)
plt.subplot(1,2,2)
plt.imshow(somim)
plt.show()