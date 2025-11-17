import torch
from micrograd import Value
import numpy as np

class Neuron:
  def __init__(self, neuron_inputs):
    self.w = [Value(np.random.uniform(-1,1)) for _ in range(neuron_inputs)]
    self.b = Value(np.random.uniform(-1,1))

  def __call__(self, x):
    # BEWARE: python's sum() by default starts with int 0, which is not a Value object
    # so you have to explicitly pass the starting value as a Value object
    # activation = sum((wi * xi for wi, xi in zip(self.w, x)), Value(0.0)) + self.b
    activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b) # another way to do the same thing
    out = activation.tanh()
    return out

  def parameters(self):
    # accumulate the parameters of the neuron, the list of knobs that can be tweaked to change the behavior of the neuron
    return self.w + [self.b]

class Layer: # a layer is a list of neurons
  def __init__(self, neuron_inputs, neuron_outputs): # neuron_inputs is the number of inputs to the layer, neuron_outputs is the number of neurons (aka outputs) in the layer
    self.neurons = [Neuron(neuron_inputs) for _ in range(neuron_outputs)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
    # another way to do the same thing
    # params = []
    # for neuron in self.neurons:
    #   ps = neuron.parameters()
    #   params.extend(ps)
    # return params

class MLP:
  def __init__(self, nin, nouts): # nin is the number of inputs to the MLP, nouts is a list of the number of neurons in EACH layer
    size = [nin] + nouts # size is a list of the number of neurons in each layer
    self.layers = [Layer(size[i], size[i+1]) for i in range(len(nouts))] # create a list of layers, each layer is a list of neurons, iterate over the pairs of adjacent neurons in the size list

  def __call__(self, x):
    for layer in self.layers: # iterate over the layers
      x = layer(x)
    return x # return the output of the MLP

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
# print("n.parameters()", n.parameters())
o = n(x)
# print(o.data)

# training data
xs = [
  [2.0, 3.0, -1.0], # desired output: 1.0, according to the desired outputs list
  [3.0, -1.0, 0.5], # desired output: -1.0
  [0.5, 1.0, 1.0], # desired output: -1.0
  [1.0, 1.0, -1.0], # desired output: 1.0
] # input data
ys = [1.0, -1.0, -1.0, 1.0] # desired outputs

# let's do many steps at once
for k in range(20):
  # this is the forward pass
  ypred = [n(x) for x in xs] # predicted outputs
  # print(ypred)
  # loss function: MSE
  loss = sum(((yp - yt)**2 for yt, yp in zip(ys, ypred)), Value(0.0))

  # now backward pass
  # first we have to reset the gradients
  # because the gradients are accumulated (not just set), we need to reset them to 0 before each backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()
  # print(n.layers[0].neurons[0].w[0].grad)

  # now the update, the gradient descent
  for p in n.parameters():
    # gradient is pointing downhill, so we need to move in the opposite direction to MINIMIZE the loss
    p.data += -0.05 * p.grad
  print(f"step {k}: loss", loss)

# ypred = [n(x) for x in xs]
# loss = sum(((yp - yt)**2 for yt, yp in zip(ys, ypred)), Value(0.0))
# print("second loss", loss)