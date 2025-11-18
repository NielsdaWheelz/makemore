from rich import print
import torch
import torch.nn.functional as F
from pathlib import Path
import random
import matplotlib.pyplot as plt

names_path = Path(__file__).with_name("names.txt")
names = names_path.read_text().splitlines()

characters = sorted(list(set(''.join(names)))) # list of unique characters in the dataset
stoi = {} # dictionary mapping characters to indices
for i, s in enumerate(characters): # enumerate returns a tuple of the index and the character
    stoi[s] = i + 1
stoi['.'] = 0 # '.' is the padding character
itos = {i:s for s,i in stoi.items()} # dictionary mapping indices to characters

block_size = 3 # the number of characters in the context window, the number of previous characters to use to predict the next character

def build_dataset(names):
  X, Y = [], [] # X: list of context windows, the 3 previous characters; Y: list of labels, the correct next character
  for name in names:
    context = [0] * block_size # initialize the context window with 0s, the '.' padding character
    for char in name + '.': # iterate over the characters in the name, plus the '.' padding character for the end of the name
      index = stoi[char] # get the index of the character
      X.append(context) # append the context window to X
      Y.append(index) # append the label to Y
      context = context[1:] + [index] # update the context window by removing the first character and adding the new character
  X = torch.tensor(X) # convert the list of context windows to a tensor
  Y = torch.tensor(Y) # convert the list of labels to a tensor
  return X, Y

random.seed(42)
random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))
Xtr, Ytr = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xte, Yte = build_dataset(names[n2:])

generator = torch.Generator().manual_seed(2147483647)
n_characters = len(itos) # the number of unique characters in the dataset
n_embeddings = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the mlp

C = torch.randn((n_characters, n_embeddings), generator = generator)
W1 = torch.randn((n_embeddings * block_size, n_hidden), generator = generator)
b1 = torch.randn((n_hidden), generator = generator)
W2 = torch.randn((n_hidden, n_characters), generator = generator) * 0.01 # initialize the weights of the output layer to small values to prevent the output layer from saturating
b2 = torch.randn(n_characters, generator = generator) * 0 # initialize the biases of the output layer to 0 to prevent the output layer from saturating
parameters = [C, W1, b1, W2, b2]
for p in parameters:
  p.requires_grad = True

max_steps = 200000 # the maximum number of steps to train for
batch_size = 32 # the number of examples to train on at a time
losses = [] # the losses for each step

for i in range(max_steps):
  # construct a mini-batch
  index = torch.randint(0, Xtr.shape[0], (batch_size,), generator = generator) # (batch_size,) a tensor of batch_size random integers between 0 and the number of examples in the training set
  Xb, Yb = Xtr[index], Ytr[index] # (batch_size, block_size) selects a tensor of batch_size examples, each with a context of block_size characters, and (batch_size,) selects a tensor of batch_size labels, each corresponding to the correct next character

  # forward pass
  embeddings = C[Xb] # EMBED CHARACTERS INTO VECTORS: (batch_size, block_size, n_embeddings) embeds the context of block_size characters into a n_embeddings-dimensional vector
  concatenated_embeddings = embeddings.view(embeddings.shape[0], -1) # CONCATENATE EMBEDDINGS: (batch_size, n_embeddings * block_size) concatenates the embeddings of the block_size characters into a single vector
  hidden_layer_pre_activation = concatenated_embeddings @ W1 + b1 # HIDDEN LAYER PRE-ACTIVATION: (batch_size, n_hidden) applies the weights and biases to the concatenated embeddings to get the hidden layer pre-activations
  h = torch.tanh(hidden_layer_pre_activation) # HIDDEN LAYER: (batch_size, n_hidden) applies the tanh activation function to the hidden layer pre-activations to get the hidden layer activations
  logits = h @ W2 + b2 # OUTPUT LAYER: (batch_size, n_characters) applies the weights and biases to the hidden layer activations to get the logits
  loss = F.cross_entropy(logits, Yb)

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  learning_rate = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -learning_rate * p.grad
  
  if i % 10000 == 0:
    print(f'{i:7d}/{max_steps:7d} loss {loss.item():.4f}')
  losses.append(loss.log10().item())

plt.plot(range(max_steps), losses)
# plt.figure(figsize=(8, 8))
# plt.scatter(C[:,0].data, C[:,1].data, s=200)
# for i in range(C.shape[0]):
#   plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="black")
# plt.grid('minor')
# plt.show()

# split losses into training, validation, and testing losses
@torch.no_grad()
def split_loss(split):
  x,y = {
    'training': (Xtr, Ytr),
    'validation': (Xdev, Ydev),
    'testing': (Xte, Yte),
  }[split]
  embeddings = C[x] # (batch_size, block_size, n_embeddings) embeds the context of block_size characters into a n_embeddings-dimensional vector
  concatenated_embeddings = embeddings.view(embeddings.shape[0], -1) # CONCATENATE EMBEDDINGS: (batch_size, n_embeddings * block_size) concatenates the embeddings of the block_size characters into a single vector
  hidden_layer_pre_activation = concatenated_embeddings @ W1 + b1 # HIDDEN LAYER PRE-ACTIVATION: (batch_size, n_hidden) applies the weights and biases to the concatenated embeddings to get the hidden layer pre-activations
  hidden_layer_activations = torch.tanh(hidden_layer_pre_activation) # HIDDEN LAYER: (batch_size, n_hidden) applies the tanh activation function to the hidden layer pre-activations to get the hidden layer activations
  logits = hidden_layer_activations @ W2 + b2 # OUTPUT LAYER: (batch_size, n_characters) applies the weights and biases to the hidden layer activations to get the logits
  loss = F.cross_entropy(logits, y) # (batch_size,) calculates the loss
  print(split, loss.item())

split_loss('training')
split_loss('validation')
split_loss('testing')

# sample
generator = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
  out = []
  context = [0] * block_size
  while True:
    embeddings = C[torch.tensor([context])] # usually we're working with the size of the training set, but here we're working with a single example, so '1' is the batch size (1, block_size, embedding_dim)
    hidden_layer_pre_activation = embeddings.view(1, -1) @ W1 + b1 # HIDDEN LAYER PRE-ACTIVATION: (1, n_hidden) applies the weights and biases to the concatenated embeddings to get the hidden layer pre-activations
    hidden_layer_activations = torch.tanh(hidden_layer_pre_activation) # HIDDEN LAYER: (1, n_hidden) applies the tanh activation function to the hidden layer pre-activations to get the hidden layer activations
    logits = hidden_layer_activations @ W2 + b2 # OUTPUT LAYER: (1, n_characters) applies the weights and biases to the hidden layer activations to get the logits
    probabilities = F.softmax(logits, dim=1)
    i = torch.multinomial(probabilities, num_samples = 1, generator = generator).item()
    context = context[1:] + [i]
    out.append(i)
    if i == 0:
      break
  print(''.join(itos[i] for i in out))