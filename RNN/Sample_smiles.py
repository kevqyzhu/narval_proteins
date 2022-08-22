#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
from Model import CharRNN
from train import one_hot_encode
import numpy as np
import torch.nn.functional as F
import argparse
import selfies as sf


def predict(net, char, h=None, top_k=None):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
    '''

    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    train_on_gpu = torch.cuda.is_available()
    if(train_on_gpu):
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    p = F.softmax(out, dim=1).data
    if(train_on_gpu):
        p = p.cpu() # move to cpu

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h


# Declaring a method to generate new text
def sample(net, size, prime='The', top_k=None):
    train_on_gpu = torch.cuda.is_available()
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()

    net.eval() # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in sf.split_selfies(prime)]
    h = net.init_hidden(1)
    for ch in sf.split_selfies(prime):
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)
    # Generating new text


parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)
parser.add_argument('--tokens', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--n_hidden', type=int, default=56)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.001)

args = parser.parse_args()

n_hidden = args.n_hidden
n_layers = args.n_layers

with open(args.tokens, "r") as f:
    chars = tuple(f.read().split('\n')[:-1])

chars = chars.__add__(tuple("\n"))
chars = chars.__add__(tuple("\00"))
# print(chars)

net = CharRNN(chars, n_hidden, n_layers, drop_prob=args.drop_prob, lr=args.lr)
net.load_state_dict(torch.load(args.model))

selfies = sample(net, args.batch_size, prime='[C]', top_k=5).split('\n')
for item in selfies:
    print(sf.decoder(item))
