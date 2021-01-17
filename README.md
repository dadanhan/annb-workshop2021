# annb-workshop2021
Docs for the ANNB workshop 2021

This is a basic workshop showing you how to:
1. setup Python3 and Tensorflow
2. create and train basic supervised neural networks in Tensorflow on images
3. create and train basic unsupervised neural networks from scratch on images
4. create and train basic supervised neural networks in Tensorflow on time series and apply this to tracking data of endosomes

Let's begin with installation.

## 1. Setup
You can download the latest version of Python3 for different operating systems [here](https://www.python.org/downloads/). Make sure you use Python **3.8.7** and NOT Python 3.9.1 as there seems to be a problem with the latest release and Tensorflow.

Download it and install Python3 on your computer. Alternatively if you want to work online, you can setup a AWS cloud account [here](https://aws.amazon.com/console/) or Google cloud account [here](https://cloud.google.com/) and work on a Cloud Shell (basically a computer maintained by them but operated by you online).

Help during installation can be found online.
(e.g. [here](https://realpython.com/installing-python/#how-to-install-python-on-windows))

To install Tensorflow on your computer, make sure that Python3 is installed and it is as simple as writing in the terminal/shell/PowerShell the following code:

```python3
pip install tensorflow
```

It usually is that simple! (Trust me, it was a lot harder before)...Detailed installation instructions for Tensorflow can be found [here](https://www.tensorflow.org/install).

To write our code and save them as .py scripts, you will need a text editor. I strongly recommend [Visual Studio Code](https://code.visualstudio.com/). Alternatively, you can use any text editor.

## 2. Basic supervised neural network

There is a plethora of information/documentation/tutorials online. So much so that it actually serves as a barrier to entry in some cases. Here we will go through a simplified tutorial of a simple, densely-connected supervised neural network to classify images. Consider this as a 'hello world' tutorial to Tensorflow. (The tutorial I based this off can be found [here](https://www.tensorflow.org/tutorials/keras/classification))

The entire code is saved in the basic_supervised.py file. We will go through this line by line in the tutorial.

#### Key concepts behind supervised neural networks

A neural network is just a mathematical operation. In essence, a single neuron with an input <img src="https://render.githubusercontent.com/render/math?math=x"> estimates some desired quantity <img src="https://render.githubusercontent.com/render/math?math=y"> using weights <img src="https://render.githubusercontent.com/render/math?math=w"> and bias <img src="https://render.githubusercontent.com/render/math?math=b">. To do this, we use a simple linear function:

<img src="https://render.githubusercontent.com/render/math?math=y = wx %2B b">

To extend this idea, now consider <img src="https://render.githubusercontent.com/render/math?math=n"> inputs <img src="https://render.githubusercontent.com/render/math?math=(x_1,x_2,...,x_n)"> to the neuron that still estimates some desired quantity <img src="https://render.githubusercontent.com/render/math?math=y">. Since we have <img src="https://render.githubusercontent.com/render/math?math=n"> inputs, we must also have a corresponding number of weights <img src="https://render.githubusercontent.com/render/math?math=(w_1,w_2,...,w_n)">. Then the simple linear function from before becomes:

<img src="https://render.githubusercontent.com/render/math?math=y=\sum_i w_i x_i %2B b">

The next natural extension to this is to have several neurons forming a 'layer' and then adding 'deep' layers which use the outputs of a layer of neurons as inputs into a subsequent layer of neurons. This can be expanded to infinity (or the power of your computer).

Now, how do we 'train' this network. In other words, how do we determine the values of the weights <img src="https://render.githubusercontent.com/render/math?math=(w_1,w_2,...,w_n)"> so that our neural network gives us the predictions <img src="https://render.githubusercontent.com/render/math?math=y"> from an input of <img src="https://render.githubusercontent.com/render/math?math=(x_1,x_2,...,x_n)">? How do we change the weights given we can compare our prediction <img src="https://render.githubusercontent.com/render/math?math=y"> and the true value <img src="https://render.githubusercontent.com/render/math?math=t">?

To do this, we need a cost function that measures the deviation of the estimated value <img src="https://render.githubusercontent.com/render/math?math=y"> from the true value <img src="https://render.githubusercontent.com/render/math?math=t">. For simplicity, we will just define it as the squared error <img src="https://render.githubusercontent.com/render/math?math=C = (y%2Dt)^2"> but you can define the cost function differently. Since the cost function is a function of <img src="https://render.githubusercontent.com/render/math?math=y">, it also depends on the weights and bias. For the simplest case above, we would update the weight and bias by

<img src="https://render.githubusercontent.com/render/math?math=w_{new} = w - r \frac{\partial C}{\partial w} ">
<img src="https://render.githubusercontent.com/render/math?math=b_{new} = b - r \frac{\partial C}{\partial b} ">

Here, <img src="https://render.githubusercontent.com/render/math?math=r "> is the learning rate. We repeatedly perform these weight and bias modifications until a certain criteria (termination condition) is met.

## 3. Basic unsupervised neural network: Self-Organized Map (SOM)

For this code, you will need to install OpenCV. You can do this by typing in your terminal/shell/PowerShell:

```python3
pip install opencv-python
```
The entire code for a SOM is saved in the basic_unsupervised_som.py file. In addition, you will need the hannahperkins_er.tif and the hannahperkins_rab5.tif files, which will be analysed/classified into neuron groups by our SOM. We will go through this code, line by line, in the tutorial.

#### Key concepts behind SOMs



## 4. Using a supervised neural network for trajectory analysis

For this section, you will first need to install the Python3 package 'stochastic' and 'pandas' by using the command:
```python
pip install stochastic
```
followed by
```python
pip install pandas
```
For documentation on the 'stochastic' package see [here](https://stochastic.readthedocs.io/en/stable/).

The complete code that you will need to generate trained neural networks is in 'supervised_nn_fbm_train.py' and the code to plot the prediction results are in 'supervised_nn_fbm_plotresults.py'.

The associated eLife paper to this neural network can be found [here](https://elifesciences.org/articles/52224)
