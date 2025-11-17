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

generator = torch.Generator().manual_seed(2147483647)

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

C = torch.randn((27, 10), generator = generator)
W1 = torch.randn((30, 200), generator = generator)
b1 = torch.randn((200), generator = generator)
W2 = torch.randn((200, 27), generator = generator)
b2 = torch.randn(27, generator = generator)
parameters = [C, W1, b1, W2, b2]

n_params = sum(p.numel() for p in parameters)
print(f"Number of parameters: {n_params}")


for p in parameters:
  p.requires_grad = True

learning_rate_exponent = torch.linspace(-3, 0, 1000) # 1000 steps, from 0.001 to 1
learning_rates = 10**learning_rate_exponent
learning_rates_used = [] # used learning rates
losses = [] # losses for each step
steps = [] # steps

for k in range(200000):
  # construct a mini-batch
  index = torch.randint(0, Xtr.shape[0], (32,)) # (32,) a tensor of 32 random integers between 0 and the number of examples in the training set

  # forward pass
  Xb = Xtr[index] # (32, 3) selects a tensor of 32 examples, each with a context of 3 characters
  Yb = Ytr[index] # (32,) selects a tensor of 32 labels, each corresponding to the correct next character
  embeddings = C[Xb] # (32, 3, 2) embeds the context of 3 characters into a 2-dimensional vector
  hidden_layer_activations = torch.tanh(embeddings.view(-1, 30) @ W1 + b1) # (32, 100) applies the weights and biases to the embeddings to get the hidden layer activations
  logits = hidden_layer_activations @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Yb)

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update the weights
  learning_rate = 0.1 if k < 100000 else 0.01
  for p in parameters:
    p.data += -learning_rate * p.grad
  
  # learning_rates_used.append(learning_rate_exponent[k])
  losses.append(loss.log10().item())
  steps.append(k)
print("training loss", loss.item())

# plt.plot(steps, losses)
# plt.figure(figsize=(8, 8))
# plt.scatter(C[:,0].data, C[:,1].data, s=200)
# for i in range(C.shape[0]):
#   plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="black")
# plt.grid('minor')
# plt.show()

# dev set evaluation
embeddings = C[Xdev] # (32, 3, 2)
h = torch.tanh(embeddings.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev) # (32,)
print("dev set evaluation", loss.item())

# test set evaluation
embeddings = C[Xte] # (32, 3, 2)
h = torch.tanh(embeddings.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Yte) # (32,)
print("test set evaluation", loss.item())

# KNOBS 
# number of neurons in each hidden layer
# dimensionality of the embeddings in the lookup table
# number of characters in the context
# optimization details: how many steps, learning rate, change in learning rate
# batch size

# sample
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
  out = []
  context = [0] * block_size
  while True:
    emb = C[torch.tensor([context])] # usually we're working with the size of the training set, but here we're working with a single example, so '1' is the batch size (1, block_size, embedding_dim)
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    i = torch.multinomial(probs, num_samples = 1, generator = g).item()
    context = context[1:] + [i]
    out.append(i)
    if i == 0:
      break
  print(''.join(itos[i] for i in out))