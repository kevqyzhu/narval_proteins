import matplotlib.pyplot as plt
from rdkit import Chem
import random
import os

molecule = "combined"
data = os.path.join(os.path.dirname(__file__), f"../../data/")
with open(f"{data}" + f"{molecule}.txt", "r") as f:
    train_smiles = f.read().split('\n')[:-1]

train_smiles = random.sample(train_smiles, 10000)

def num_heavy_plot(train_smiles):
    qed_train = []

    m2 = 0
    for mol in train_smiles:
        if Chem.MolFromSmiles(mol) is not None:
            num = Chem.MolFromSmiles(mol).GetNumHeavyAtoms()
            if num > m2:
                m2 = num
            qed_train.append(num)

    fig = plt.figure()
    ax2 = fig.add_subplot()

    ax2.hist(qed_train, bins=list(range(m2)))

    ax2.set_title(f'Training {molecule}')
    ax2.set_xlabel('Number of Heavy Atoms')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f"../{molecule}/hist_num_heavy.png"))


num_heavy_plot(train_smiles)









