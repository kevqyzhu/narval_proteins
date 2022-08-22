import numpy as np
import selfies as sf
import os

mol = "all-fragments"

data = os.path.join(os.path.dirname(__file__), f"../data/")

with open(data + f"{mol}_selfies.txt", "r") as f:
    text = f.read().split('\n')[:-1]

chars = tuple(sf.get_alphabet_from_selfies(text))
print(chars)
chars = chars.__add__(tuple("\n"))
char_vec_length = len(chars)


longest = 0
total = 0
for mol in text:
    total += len(list(sf.split_selfies(mol)))
    if longest <= len(list(sf.split_selfies(mol))):
        longest = len(list(sf.split_selfies(mol)))
longest += 1
print(f"The average is: {total/len(text)}")
print(f"The longest is: {longest}")

# hidden_nodes = int(2/3 * (word_vec_length * char_vec_length))
hidden_nodes = int(2/3 * (longest * char_vec_length))
print(f"The number of hidden nodes is {hidden_nodes}.")


