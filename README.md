# annb-workshop2021
Docs for the ANNB workshop 2021

This is a basic workshop showing you how to:
1. setup Python3 and tensorflow
2. create and train basic supervised neural networks in tensorflow on images
3. create and train basic unsupervised neural networks from scratch on images
4. create and train basic supervised neural networks in tensorflow on time series and apply this to tracking data of endosomes

Let's begin with installation.

## 1. Setup
You can download the latest version of Python3 for different operating systems [here](https://www.python.org/downloads/).

Download it and install Python3 on your computer. Alternatively if you want to work online, you can setup a AWS cloud account [here](https://aws.amazon.com/console/) or Google cloud account [here](https://cloud.google.com/) and work on a Cloud Shell (basically a computer maintained by Google but operated by you online).

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

But this is boring, it's just a linear function. To form a useful neural network, we need to have multiple neurons working in parallel with a series of inputs.


We can extend this by adding 'deep' layers which use the outputs of a layer of neurons as inputs into a subsequent layer of neurons. This can be expanded to infinity (or the power of your computer).

## 3. Basic unsupervised neural network: Self-Organized Map (SOM)

## 4. Using a supervised neural network for trajectory analysis

The associated eLife paper to this neural network can be found [here](https://elifesciences.org/articles/52224)

Sample equation
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
