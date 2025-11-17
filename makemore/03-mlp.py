from rich import print
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

names_path = Path(__file__).with_name("names.txt")
names = names_path.read_text().splitlines()

characters = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(characters)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

generator = torch.Generator().manual_seed(2147483647)

block_size = 3 # context length, number of characters to look at to predict the next character
X, Y = [], [] # inputs and labels

for name in names:
  context = [0] * block_size # start with a context of all 0s
  for char in name + '.': # iterate over the characters in the name. padding with '.' to the end
    index = stoi[char] # get the index of the character
    X.append(context) # add the context to the inputs
    Y.append(index) # add the index to the labels
    context = context[1:] + [index] # shift the context to the right
X = torch.tensor(X)
Y = torch.tensor(Y)

# each character will have a 2-dimensional embedding
C = torch.randn((27, 2), generator = generator)
# embedding_for_e = C[5] # embedding for the character 'e'
# one_hot_for_e = F.one_hot(torch.tensor([5]), num_classes = 27).float() # one-hot encoding for the character 'e'
# embedding_for_e == one_hot_for_e @ C # embedding for the character 'e'
W1 = torch.randn((6, 100), generator = generator) # 3x2=6 is the number of neurons in the first layer, 100 is the number of neurons in the second layer
b1 = torch.randn((100), generator = generator) # bias for the first layer
W2 = torch.randn((100, 27), generator = generator) # 100 is the number of neurons in the second layer, 27 is the number of characters
b2 = torch.randn(27, generator = generator) # bias for the second layer
parameters = [C, W1, b1, W2, b2]

for p in parameters:
  p.requires_grad = True

# learning_rate_exponent = torch.linspace(-3, 0, 1000) # 1000 steps, from 0.001 to 1
# learning_rates = 10**learning_rate_exponent

# learning_rates_used = []
# losses = []
for k in range(10000):
  # create mini-batches
  # more less-accurate steps is more efficient than less and more accurate steps, because it's faster to compute the gradient and update the weights
  index = torch.randint(0, X.shape[0], (32,)) # index is a tensor of shape (32,), where the first dimension is the number of examples, the second dimension is the index of the example
  Xb = X[index] # Xb is a tensor of shape (32, 3), where the first dimension is the number of examples, the second dimension is the index of the example
  Yb = Y[index] # Yb is a tensor of shape (32,), where the first dimension is the number of examples, the second dimension is the index of the example
  # forward pass
  # C is an embedding matrix — a lookup table that turns character indices into small vectors of numbers.
  # If you have 27 possible characters (., a–z), and each embedding is 2 numbers long, then C has shape (27, 2).
  # So C[i] gives you a 2-number “meaningful representation” for character i.
  # Xb is a matrix of shape (32, 3) — 32 rows (examples), 3 columns (previous characters).
  # Now each of the 32 examples has 3 characters × 2 features each.
  embeddings = C[Xb] # embeddings for the input context. Xb(32, 3)x2(2 embedding dimensions)

  # Current shape of embeddings: (32, 3, 2). We're flattening the last two dimensions into one long vector per example.
  # Each example has 3 characters × 2 numbers = 6 numbers total. Flattened shape: (32, 6)
  # .view(-1, 6) reshapes the embeddings tensor to a 2D tensor, where the first dimension is inferred from the shape of the other dimensions (-1), and 6 is the number of neurons in the first layer
  # W1 is a weight matrix of size (6, 100). Each of the 100 “neurons” in the hidden layer looks at all 6 inputs and forms a weighted sum.
  # b1 is a bias vector of size (100,) — one bias per neuron.
  # So before activation, each of the 32 examples becomes a 100-element vector of “neuron inputs”.
  hidden_layer_activations = torch.tanh(embeddings.view(-1, 6) @ W1 + b1)
  logits = hidden_layer_activations @ W2 + b2
  # counts = logits.exp()
  # probabilities = counts / counts.sum(1, keepdim = True)
  # probabilities = F.softmax(logits, dim=1)
  # probabilities is a tensor of shape (32, 27), where the first dimension is the number of examples, the second dimension is the number of characters
  # loss = -probabilities[torch.arange(32), Y].log().mean()
  loss = F.cross_entropy(logits, Yb) # another way to calculate the loss

  # reset the gradients to 0
  for p in parameters:
    p.grad = None

  # backward pass
  loss.backward()

  # update the weights
  # learning_rate = learning_rates[k]
  learning_rate = 0.1
  # decay this learning rate over time
  # learning_rate = learning_rate / (1 + k / 1000)
  for p in parameters:
    p.data += -learning_rate * p.grad
  # learning_rates_used.append(learning_rate_exponent[k])
  # losses.append(loss.item())
  # print(f"step {k}: loss {loss.item()}")
print(loss.item())

# plt.plot(learning_rates_used, losses)
# plt.show()