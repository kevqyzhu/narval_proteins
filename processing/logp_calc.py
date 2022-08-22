import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
import random
import numpy as np
import os

molecule = "moses"
dir = "moses_outputs_seq_52_hidden_256_dp_0.3_lr_0.0001"
epoch = 134

gen_dir = os.path.join(os.path.dirname(__file__), f"../{molecule}/{dir}/")


with open(f"{gen_dir}" + f"novel_outputs{epoch}.txt", "r") as f:
    gen_smiles = f.read().split('\n')[:-1]

data = os.path.join(os.path.dirname(__file__), f"../../data/")

with open(f"{data}" + f"{molecule}.txt", "r") as f:
    train_smiles = f.read().split('\n')[:-1]

train_smiles = random.sample(train_smiles, 10000)


qed_gen = []
qed_train = []

m1 = 0
for mol in gen_smiles:
    if Chem.MolFromSmiles(mol) is not None:
        num = MolLogP(Chem.MolFromSmiles(mol))
        if num > m1:
            m1 = num
        qed_gen.append(num)

m2 = 0
for mol in train_smiles:
    if Chem.MolFromSmiles(mol) is not None:
        num = MolLogP(Chem.MolFromSmiles(mol))
        if num > m2:
            m2 = num
        qed_train.append(num)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.hist(qed_gen, bins=list(np.arange(0, m1, 0.1)))
ax2.hist(qed_train, bins=list(np.arange(0, m2, 0.1)))

ax1.set_title(f'Generated {molecule}')
ax1.set_xlabel('LogP')
ax1.set_ylabel('Frequency')

ax2.set_title(f'Training {molecule}')
ax2.set_xlabel('LogP')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(gen_dir+f'hist_logp_{epoch}.png')









