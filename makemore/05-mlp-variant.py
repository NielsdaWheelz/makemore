from rich import print
import torch
import torch.nn.functional as F
from pathlib import Path
import random
import matplotlib.pyplot as plt

names_path = Path(__file__).with_name("names.txt")
names = names_path.read_text().splitlines()

characters = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(characters)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 3

def build_dataset(names):
  X, Y = [], []
  for name in names:
    context = [0] * block_size
    for char in name + '.':
      index = stoi[char]
      X.append(context)
      Y.append(index)
      context = context[1:] + [index]
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

random.seed(42)
random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))
Xtr, Ytr = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xte, Yte = build_dataset(names[n2:])

generator = torch.Generator().manual_seed(2147483647)
n_characters = len(itos) # the number of characters in the dataset
n_embeddings = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the mlp

C = torch.randn((n_characters, n_embeddings), generator = generator)
W1 = torch.randn((n_embeddings * block_size, n_hidden), generator = generator)
b1 = torch.randn((n_hidden), generator = generator)
W2 = torch.randn((n_hidden, n_characters), generator = generator) * 0.01
b2 = torch.randn(n_characters, generator = generator) * 0
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
  h = torch.tanh(concatenated_embeddings @ W1 + b1) # HIDDEN LAYER: (batch_size, n_hidden) applies the weights and biases to the concatenated embeddings to get the hidden layer activations
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

# split losses into training, development, and test losses
@torch.no_grad()
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'validation': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  embeddings = C[x] # (batch_size, block_size, n_embeddings) embeds the context of block_size characters into a n_embeddings-dimensional vector
  concatenated_embeddings = embeddings.view(embeddings.shape[0], -1) # CONCATENATE EMBEDDINGS: (batch_size, n_embeddings * block_size) concatenates the embeddings of the block_size characters into a single vector
  h = torch.tanh(concatenated_embeddings @ W1 + b1) # (batch_size, n_hidden) applies the weights and biases to the concatenated embeddings to get the hidden layer activations
  logits = h @ W2 + b2 # (batch_size, n_characters) applies the weights and biases to the hidden layer activations to get the logits
  loss = F.cross_entropy(logits, y) # (batch_size,) calculates the loss
  print(split, loss.item())

split_loss('train')
split_loss('validation')
split_loss('test')

# sample
g = torch.Generator().manual_seed(2147483647 + 10)

# for _ in range(20):
#   out = []
#   context = [0] * block_size
#   while True:
#     emb = C[torch.tensor([context])] # usually we're working with the size of the training set, but here we're working with a single example, so '1' is the batch size (1, block_size, embedding_dim)
#     h = torch.tanh(emb.view(1, -1) @ W1 + b1)
#     logits = h @ W2 + b2
#     probs = F.softmax(logits, dim=1)
#     i = torch.multinomial(probs, num_samples = 1, generator = g).item()
#     context = context[1:] + [i]
#     out.append(i)
#     if i == 0:
#       break
#   print(''.join(itos[i] for i in out))