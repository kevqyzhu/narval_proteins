import numpy as np
import os
import matplotlib.pyplot as plt

mol = "all-fragments"
data = os.path.join(os.path.dirname(__file__), f"../data/")


with open(data + f"{mol}.txt", "r") as f:
    text = f.read()

chars = list(set(text))
chars.sort()
chars = tuple(chars)
print(chars)
char_vec_length = len(chars)


with open(data + f"{mol}.txt", "r") as f:
    text = f.read().split('\n')[:-1]

longest = 0
total = 0
length_list = []
for mol in text:
    # total += len(list(sf.split_selfies(mol)))
    if longest <= len(mol):
        longest = len(mol)
        # print(sf.decoder(mol))
        # print(f"longest:{longest + 1}")
    length_list.append(len(mol))

longest += 1
# print(f"The average is: {total/len(text)}")
print(f"The longest is: {longest}")

# hidden_nodes = int(2/3 * (word_vec_length * char_vec_length))
# hidden_nodes = int(2/3 * (longest * char_vec_length))
# print(f"The number of hidden nodes is {hidden_nodes}.")


