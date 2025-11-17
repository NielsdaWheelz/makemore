from rich import print
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
# open('names.txt') in backend/makemore/makemore.py:1 looks in whatever directory you launch the script from. 
# When you run it from the repo root, there is no names.txt, so Python raises a FileNotFoundError.
names_path = Path(__file__).with_name("names.txt")
names = names_path.read_text().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

characters = sorted(list(set(''.join(names))))
string_to_int = {string:integer+1 for integer, string in enumerate(characters)}
string_to_int['.'] = 0
int_to_string = {integer:string for string, integer in string_to_int.items()}

# char_counts = sorted(b.items(), key = lambda key_value: key_value[1], reverse=True)

# count the bigrams in the names
for n in names:
  chars = ['.'] + list(n) + ['.']
  # zip returns tuples
  # but if one array is shorter, zip will stop at the end of the shorter array
  for char1, char2 in zip(chars, chars[1:]):
    integer_1 = string_to_int[char1]
    integer_2 = string_to_int[char2]
    bigram = (integer_1, integer_2)
    N[integer_1, integer_2] += 1
    # print(char1, char2)

# print(int_to_string)

# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap="Blues")
# for i in range(27):
#   for j in range(27):
#     char_string = int_to_string[i] + int_to_string[j]
#     plt.text(j, i, char_string, ha="center", va="bottom", color="gray")
#     plt.text(j, i, N[i,j].item(), ha="center", va="top", color="gray")
# plt.axis("off")
# plt.show()

# print(N[0])

# probability of a given first character
# turn the row into a probability distribution by dividing by the sum of the row
p = N[0].float() # turn the row into a float tensor
p = p / p.sum()

# randomly initialize 27 neurons' weights. each neuron receives 27 inputs and produces 1 output
g = torch.Generator().manual_seed(2147483647) # seed the generator so i match karpathy
# p = torch.rand(3, generator=g) # generate 3 random numbers from the probability distribution
# p = p / p.sum()

# use the multinomial to draw samples from it
multinomial = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
# letters = [int_to_string[i.item()] for i in multinomial]
# print(''.join(letters))
letter = int_to_string[multinomial.item()]
# print(letter)

# probabilities; basically our parameters
P = (N+1).float()
P /= P.sum(1, keepdim=True) # divide each row by the sum of the row to get a probability distribution
# print(P)

for i in range(10):
  out = []
  index = 0
  while True:
    p = P[index]
    # p = N[index].float() # turn the row into a float tensor
    # p = p / p.sum() # turn the row into a probability distribution
    index = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # draw a sample from the probability distribution
    out.append(int_to_string[index])
    if index == 0: # if the sample is 0, we have reached the end of the name
      break
  # print(''.join(out))

for n in names[:3]:
  chars = ['.'] + list(n) + ['.']
  # zip returns tuples
  # but if one array is shorter, zip will stop at the end of the shorter array
  log_likelihood = 0.0
  n = 0
  for char1, char2 in zip(chars, chars[1:]):
    integer_1 = string_to_int[char1]
    integer_2 = string_to_int[char2]
    p = P[integer_1, integer_2]
    log_p = torch.log(p)
    log_likelihood += log_p
    n += 1
avg_log_likelihood = -log_likelihood/n
    # print(char1, char2, p.item(), log_p.item())
# -log_likelihood/n is the average log probability of the characters in the name
# print(f'{-log_likelihood/n=}')

# NEURAL NETWORK
# create training set of bigrams: (input{x}, output{y})
xs, ys = [], []
for n in names[:1]:
  chars = ['.'] + list(n) + ['.']
  # zip returns tuples
  # but if one array is shorter, zip will stop at the end of the shorter array
  for char1, char2 in zip(chars, chars[1:]):
    integer_1 = string_to_int[char1]
    integer_2 = string_to_int[char2]
    xs.append(integer_1)
    ys.append(integer_2)

# now create tensors out of these lists
xs = torch.tensor(xs)
ys = torch.tensor(ys)

# initialize weights
# randomly initialize 27 neurons' weights. each neuron receives 27 inputs and produces 1 output
g = torch.Generator().manual_seed(2147483647) # seed the generator so i match karpathy
weights = torch.randn((27, 27), generator=g, requires_grad=True)
# each column in W is one neuron that votes for a particular next character

# FORWARD PASS
# one-hot encoding for alphabet (binary encoding)
x_encoded = F.one_hot(xs, num_classes=27).float()

# plt.figure(figsize=(16,16))
# plt.imshow(x_encoded, cmap="Blues")
# plt.show()

# muiltiply weights by inputs to get predictions/outputs
# matrix multiplication: (27, 27) @ (27, 27) = (27, 27); where '@' is matrix multiplication in pytorch
logits = x_encoded @ weights
# Because x is one-hot, x @ W literally selects the row of W corresponding to the input letter

# output is a tensor of shape (27, 27); the first dimension is the number of characters, the second dimension is the number of neurons
# effectively this is the 27 activations for each neuron
# (x_encoded @ W)[3, 13] = x_encoded[3] * W[:, 13] is the firing rate of the 13th neuron for the 3rd input

# predict log-counts
counts = logits.exp() # equivalent to the 'N' from before
probabilities = counts / counts.sum(1, keepdims=True) # probabilities for each character, given the previous character
# probability distribution for the next character
# these last two lines are equivalent to the 'P' from before, AKA 'softmax'
# Softmax: takes logits and converts them to probabilities; the probabilities are normalized to sum to 1

# For every training example (every (prev → next) pair), we have a 27-length probability vector that says how likely each next character is according to our current weights

# for each of our bigrams, we have a row of probabilities, normalized to sum to 1
# print(p[0].sum(), p.shape)

# nlls = torch.zeros(5)
# for i in range(5):
#   x = xs[i].item()
#   y = ys[i].item()
#   p = probabilities[x, y]
#   logp = torch.log(p)
#   nll = -logp
#   nlls[i] = nll
# print

# calc loss: but not mse (because that's for regression), but negative log likelihood (because that's for classification)
# probabilities[torch.arange(5), ys] is the probability of the correct next character for each of the 5 training examples
# .log() is the log of the probability
# .mean() is the mean of the log probabilities
loss = -probabilities[torch.arange(5), ys].log().mean()
# done forward pass

# now backward pass
# we want to get the gradient of the loss with respect to the weights
# we use the chain rule to do this
weights.grad = None # reset the gradients to 0
loss.backward()
print(weights.grad)
# now we have the gradient of the loss with respect to the weights
weights.data += -0.1 * weights.grad