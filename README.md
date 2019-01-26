# Intro to PyTorch
#### CS11-747 Neural Networks for NLP
___
### What is PyTorch ?
It’s a Python-based scientific computing package targeted at two sets of audiences:
* A replacement for NumPy to use the power of GPUs
* A deep learning research platform that provides flexibility (dynamic)

Before diving into tensors, let's refresh our numpy concepts.

### Primer on Numpy
* Dealing with tensors in PyTorch is similar to dealing with matrices in numpy
* Let's begin our crash course on Numpy
* Open *numpy_tutorial.ipynb*

There is a similar tutorial for tensors, which you can see at your own time.
Refer to *tensor_tutorial.ipynb*, which was shamelessly copied from <https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html>.
[PyTorch documentation](https://pytorch.org/tutorials/) is a wonderful resource for learning how to use PyTorch. 

### Autograd: Automatic Differentiation
Central to all neural networks in PyTorch is the autograd package.
* Let’s first briefly visit this, and we will then go to training our first neural network
* The autograd package provides automatic differentiation for all operations on Tensors. It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different.

#### Tensors

* torch.Tensor is the central class of the package. If you set its attribute .requires_grad as True, it starts to track all operations on it. When you finish your computation you can call .backward() and have all the gradients computed automatically. The gradient for this tensor will be accumulated into .grad attribute.
* To stop a tensor from tracking history, you can call .detach() to detach it from the computation history, and to prevent future computation from being tracked. (Needed in some scenarios)
* Let's do a small exercise to understand how auto-differentiation works.
* Open *autograd_tutorial.ipynb*

The below diagram is a rough representation of a computational graph which will be created for the example in *autograd_tutorial.ipynb*

![alt text](https://github.com/MysteryVaibhav/pytorch_tutorial/blob/master/computation_graph.jpg "Computation Graph Example")

### Multi Layer Perceptron
I am assuming everyone has heard of multi-layer perceptrons / feed-forward networks.
* Let's train a single hidden layer feed forward network for digit classification.
* Dataset ? Obviously MNIST !!
* Time to open *mlp_tutorial.ipynb*

### A glipmse into an actual code
*lstm_tutorial/* contains an example for doing text classification. The code is organized into different classes and might be beneficial for people who have never written a PyTorch code before.

* The dynet equivalent of the code is available at <https://github.com/neubig/nn4nlp-code/blob/master/06-rnn/sentiment-lstm.py>. However this version has some extra stuff like mini-batching, loading pre-trained embeddings and dropouts.
* Open *lstm_tutorial/main.py*
