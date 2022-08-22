import numpy as np
import torch
torch.cuda.empty_cache()

from train import train
from Model import CharRNN
import argparse
import selfies as sf
from random import shuffle
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--tokens', required=True)
parser.add_argument('--encoded', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_epoch', type=int, default=-1)
parser.add_argument('--n_epoch', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_length', type=int, default=50)
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_hidden', type=int, default=56)
parser.add_argument('--n_layers', type=int, default=2)
args = parser.parse_args()

# Declaring the hyperparameters
batch_size = args.batch_size
n_epochs = args.n_epoch
seq_length = args.seq_length
load_epoch = args.load_epoch
learning_rate = args.lr

n_hidden = args.n_hidden
n_layers = args.n_layers

with open(args.tokens, "r") as f:
    chars = tuple(f.read().split('\n')[:-1])
chars = chars.__add__(tuple("\n"))

net = CharRNN(chars, n_hidden, n_layers, drop_prob=args.drop_prob, lr=args.lr)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

if args.load_epoch >= 0:
    checkpoint = torch.load(args.save_dir + "/" + f'rnn_{args.load_epoch}_epoch.net')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(net)
with open(args.save_dir + "/model_size.txt",
          "w") as f:
    f.write(str(net))
    f.write(": ")
    f.write(str(sum([x.nelement() for x in net.parameters()])))

# Declaring the hyperparameters
batch_size = args.batch_size
seq_length = args.seq_length
n_epochs = args.n_epoch
# start smaller if you are just testing initial behavior

# total = sum([x.nelement() for x in net.parameters()])
# with open('/home/kevqyzhu/scratch/proteins/params.txt', 'a') as f:
#     f.write(f"{net}: {total} \n")
# quit() 

if os.path.isfile(args.encoded):
    encoded = pickle.load(open(args.encoded, 'rb'))

# print(chars)
# print(encoded[:40])

# train the model

train(net, args.save_dir, encoded, load_epoch=args.load_epoch, epochs=n_epochs,
      batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10000)






