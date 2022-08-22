import selfies as sf
from random import shuffle
import os

mol = "all-fragments"

data = os.path.join(os.path.dirname(__file__), f"../data/")

with open(data + f"{mol}_selfies.txt", "r") as f:
    text = f.read().split('\n')[:-1]
shuffle(text)
chars = tuple(sorted(sf.get_alphabet_from_selfies(text)))
print(chars)


with open(data + f"{mol}_tokens.txt", "w") as f:
    for item in chars:
        f.write("%s\n" % item)

with open(data + f"{mol}_tokens.txt", "r") as f:
    chars2 = tuple(f.read().split('\n')[:-1])

print(chars2)

chars2 = chars2.__add__(tuple("\n"))
chars2 = chars2.__add__(tuple("\00"))
int2char = dict(enumerate(chars2))
print(int2char)
char2int = {ch: ii for ii, ch in int2char.items()}
print(char2int)

# seq_length = 22
# longest = 0
# for mol in text:
#     if longest <= len(list(sf.split_selfies(mol))):
#         longest = len(list(sf.split_selfies(mol)))
#
# seq_length = max(longest, seq_length) + 1
# print(seq_length)
#
# ls = []
# for mol in text:
#     vocab = sf.split_selfies(mol)
#     for ch in vocab:
#         ls.append(char2int[ch])
#     ls.append(char2int["\n"])
#     for _ in range(seq_length-(sf.len_selfies(mol)+1)):
#         ls.append(char2int["\00"])
#
# encoded = np.array(ls)
# print(encoded)
#
# with open("/Users/kevinzhu/Documents/UTORONTO/AI_Research/genmol/genmol/data/qm9_encoded_1.txt", "w") as f:
#     for item in encoded:
#         f.write("%s\n" % item)
#
# batch_size = 6
#
# batch_size_total = batch_size * seq_length
# # total number of batches we can make
# n_batches = len(encoded)//batch_size_total
#
# # Keep only enough characters to make full batches
# encoded = encoded[:n_batches * batch_size_total]
# # Reshape into batch_size rows
# encoded = encoded.reshape((batch_size, -1))
#
# for n in range(0, encoded.shape[0], seq_length):
#     # The features
#     x = encoded[:, n:n+seq_length]
#     # The targets, shifted by one
#     y = np.zeros_like(x)
#     try:
#         y[:, :-1], y[:, -1] = x[:, 1:], encoded[:, n+seq_length]
#     except IndexError:
#         y[:, :-1], y[:, -1] = x[:, 1:], encoded[:, 0]
#     if n <= 0:
#         print(x.astype(int), y.astype(int))









