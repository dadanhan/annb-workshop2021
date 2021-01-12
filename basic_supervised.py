#Code for a basic supervised neural network that takes images of clothing and trains a neural network to classify what that image is. This is based off the tutorial available on https://www.tensorflow.org/tutorials/keras/classification [Date accessed: 12 Jan 2021].

#This is the line that imports the Tensorflow package so you can call it in your code later as 'tf'.
import tensorflow as tf
#The numpy package helps us manipulate large arrays/matrices and perform high-level mathematical operations on them.
import numpy as np
#The matplotlib package allows us to plot any results nicely.
import matplotlib.pyplot as plt

#Let's test our Tensorflow package and check it has loaded correctly. It should print out the version of Tensorflow installed on your Python3.
print(tf.__version__)

#If you get a URLError [SSL:CERTIFICATE_VERIFY_FAILED] from trying to load the dataset, uncomment lines 14 and 15 and try running the script again. It is a common problem on Apple macs.
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

#let's load the data from the mnist fashion dataset.
fashion_mnist_dataset = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_dataset.load_data()
#Lines 18 and 19 have loaded our dataset into numpy arrays, which is done automatically by the function load_data(). The prefix 'train' is going to be the data that we use to train our neural network. 'train_images' are the inputs to our neurons and 'train_labels' are the outputs that we desire our neural network to estimate. The prefix 'test' is going to be the data that we use to test our neural network and evaluate its accuracy.
#Let's print the shape of these numpy arrays and see some images
print('train images',train_images.shape,'train labels',train_labels.shape)
print('test images',test_images.shape,'test labels',test_labels.shape)
#Let's display some images and their labels
plt.figure()
for i in range(1,10):
    plt.subplot(3,3,i)
    plt.imshow(train_images[i+10],cmap=plt.cm.binary)
    plt.title(train_labels[i+10])
plt.tight_layout()
plt.show()
#The images are 28 by 28 pixel numpy arrays that contain 8-bit pixel intensity integers (0-255). The labels are integer values (0-9) indicating what type of clothing the image displays.
# 0 -> t-shirt/top
# 1 -> trousers
# 2 -> pullover
# 3 -> dress
# 4 -> coat
# 5 -> sandal
# 6 -> shirt
# 7 -> sneaker
# 8 -> bag
# 9 -> ankle boot
#We will create a list that corresponds to this order so that we can print out nicely what each item is
type_names = ['t-shirt/top', 'trousers', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#Let's display images and their string labels again
plt.figure()
for i in range(1,10):
    plt.subplot(3,3,i)
    plt.imshow(train_images[i+10],cmap=plt.cm.binary)
    plt.title(type_names[train_labels[i+10]])
plt.tight_layout()
plt.show()

#In order to preprocess the images before training the data, we will normalize the images so that, instead of values ranging from 0-255, the values will range from 0-1.
train_images = train_images / 255.0
test_images = test_images / 255.0

#Now we set up the network structure or what we call the 'model'. This is going to be a simple densly connected network.
model= tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #input the 28 by 28 pixel array as a series of (784) numbers in to the neural network
    tf.keras.layers.Dense(128, activation='relu'), #a layer densely connected to the input with 128 neurons with a rectified linear unit activation function. See https://www.tensorflow.org/api_docs/python/tf/keras/activations
    tf.keras.layers.Dense(10) #the final layer densely connected with the previous layer with 10 neurons that corresponds to a likelihood score of the 10 types of clothing. In effect, we are training the network to score how likely each type is given the corresponding image
])
#For different types of layers see https://www.tensorflow.org/api_docs/python/tf/keras/layers

#Before the we train the model, we need to add an optimizer, a loss function and a metric.
model.compile(optimizer='adam', #the optimizer tells the model how to change the model if it made a mistake in estimation. See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #the loss function measures the accuracy of the estimation. See https://keras.io/api/losses/
              metrics=['accuracy']) #the metric is a parameter that we define to monitor the 'goodness' of training and testing steps. Here we use 'accuracy' which is the fraction of correctly classified images. See https://www.tensorflow.org/api_docs/python/tf/keras/metrics

#To train the model on the training dataset, it's as simple as:
model.fit(train_images, train_labels, epochs=10)
#To evaulate the model on the test dataset:
print('Evaluate on test dataset')
test_loss, test_accuracy = model.evaluate(test_images,  test_labels)
print('Test loss =',test_loss)
print('Test accuracy = ',test_accuracy)

#Now we will use this trained neural network to predict from the test images
prediction = model.predict(test_images)
print('Numpy array of the output values (10 neurons) from the trained model')
print(prediction)
print('shape of the predictions array')
print(prediction.shape)

#To turn the outputs from the 10 neurons in the trained model (a vector with 10 elements) into probabilities which we can then use to classify the most likely clothing type, we need to use the softmax of the vector. We can use a function to do this:
def softmax(x):
    softmax_x = np.exp(x)
    return softmax_x/np.sum(softmax_x)

#plot the prediction results for the test image at index i
i = 10
print('True label',test_labels[i])
print('Softmax prediction label',np.argmax(softmax(prediction[i])))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(test_images[i],cmap=plt.cm.binary)
plt.title('Predicted: '+type_names[np.argmax(softmax(prediction[i]))]+', Label: '+type_names[test_labels[i]])
plt.subplot(1,2,2)
plt.bar(range(0,10),softmax(prediction[i]))
plt.xticks(range(0,10),type_names,rotation=90)
plt.ylabel('probability')
plt.show()