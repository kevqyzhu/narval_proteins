import torch
from Model import CharRNN
from train import one_hot_encode
import numpy as np
import torch.nn.functional as F
import argparse
import selfies as sf
import sys

sys.setrecursionlimit(100000)

class SmilesRnnSampler:
    """
    Samples molecules from an RNN smiles language model
    """

    def __init__(self, device: str, batch_size=64) -> None:
        """
        Args:
            device: cpu | cuda
            batch_size: number of concurrent samples to generate
        """
        self.device = device
        self.batch_size = batch_size

    def predict(self, net, char, h=None, top_k=None):
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
    def sample(self, net, size, prime='The', top_k=None):
        train_on_gpu = torch.cuda.is_available()
        if(train_on_gpu):
            net.cuda()
        else:
            net.cpu()

        net.eval() # eval mode

        # First off, run through the prime characters
        chars = [ch for ch in prime]
        h = net.init_hidden(1)
        for ch in prime:
            char, h = self.predict(net, ch, h, top_k=top_k)

        chars.append(char)

        # Now pass in the previous character and get a new one
        # for ii in range(size):
        #     char, h = self.predict(net, chars[-1], h, top_k=top_k)
        #     chars.append(char)

        count = 0
        while count < size:
            char, h = self.predict(net, chars[-1], h, top_k=top_k)
            chars.append(char)
            if char == '\n':
                count += 1

        return ''.join(chars)
        # Generating new text

    def sample_selfies(self, net, size, prime='The', top_k=None):
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
            char, h = self.predict(net, ch, h, top_k=top_k)

        chars.append(char)

        # Now pass in the previous character and get a new one
        # for ii in range(size):
        #     char, h = self.predict(net, chars[-1], h, top_k=top_k)
        #     chars.append(char)

        count = 0
        while count < size:
            char, h = self.predict(net, chars[-1], h, top_k=top_k)
            chars.append(char)
            if char == '\n':
                count += 1

        return ''.join(chars)
        # Generating new text

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', required=True)
parser.add_argument('--tokens', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--output_file', required=True)
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

net = CharRNN(chars, n_hidden, n_layers, drop_prob=args.drop_prob, lr=args.lr)

checkpoint = torch.load(args.model)
net.load_state_dict(checkpoint['model_state_dict'])

sampler = SmilesRnnSampler(device)

if "smiles" in args.output_file:
    prime = 'C'
    output = sampler.sample(net, args.batch_size, prime=prime, top_k=5).split('\n')
else:
    prime = '[C]'
    output = sampler.sample_selfies(net, args.batch_size, prime=prime, top_k=5).split('\n')

with open(args.output_file, 'w') as f:
    if "smiles" in args.output_file:
        for smiles in output:
            f.write(smiles + '\n')
    else:
        for selfies in output:
            try:
                smiles = sf.decoder(selfies)
            except:
                smiles = None
                print("smiles conversion failed")

            if smiles is not None:
                f.write(smiles + '\n')
            



