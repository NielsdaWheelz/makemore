from rich import print
import torch
import torch.nn.functional as F
from pathlib import Path
names_path = Path(__file__).with_name("names.txt")
names = names_path.read_text().splitlines()

characters = sorted(list(set(''.join(names))))
string_to_integer = {string:integer+1 for integer, string in enumerate(characters)}
string_to_integer['.'] = 0
integer_to_string = {integer:string for string, integer in string_to_integer.items()}

# dataset
xs, ys = [], []
for name in names:
  characters = ['.'] + list(name) + ['.']
  for character_1, character_2 in zip(characters, characters[1:]):
    index_1 = string_to_integer[character_1]
    index_2 = string_to_integer[character_2]
    xs.append(index_1)
    ys.append(index_2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
number_of_examples = xs.nelement()

generator = torch.Generator().manual_seed(2147483647)
weights = torch.randn((27, 27), generator = generator, requires_grad = True)

for k in range(100):
  # forward pass
  x_encoded = F.one_hot(xs, num_classes = 27).float() # one-hot encode the inputs
  logits = x_encoded @ weights # predict log-counts
  # counts = logits.exp() # counts; equivalent to 'N'
  # probabilities = counts / counts.sum(1, keepdim = True) # probabilities for next character, given the previous character
  probabilities = F.softmax(logits, dim=1) # probabilities for next character, given the previous character
  loss = -probabilities[torch.arange(number_of_examples), ys].log().mean() + 0.0001 * (weights**2).mean() # negative log likelihood loss
  # L2 regularization: add a small penalty to the loss to prevent overfitting - tries to keep the weights small, closer to 0
  print(f"step {k}: loss {loss.item()}")

  # backward pass
  weights.grad = None # reset the gradients to 0
  loss.backward()

  # update the weights
  weights.data += -40 * weights.grad # gradient descent

# sample from the model
for i in range(10):
  out = []
  index = 0

  while True:
    x_encoded = F.one_hot(torch.tensor([index]), num_classes = 27).float()
    logits = x_encoded @ weights
    # counts = logits.exp()
    # probabilities = counts / counts.sum(1, keepdim = True)
    probabilities = F.softmax(logits, dim=1) 

    index = torch.multinomial(probabilities, num_samples = 1, replacement = True, generator = generator).item()
    out.append(integer_to_string[index])
    if index == 0:
      break
  print(''.join(out))