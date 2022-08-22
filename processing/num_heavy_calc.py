import matplotlib.pyplot as plt
from rdkit import Chem
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mol', default="largestfragments")
parser.add_argument('--model_dir', default="largestfragments_outputs_seq_2123_hidden_700_dp_0.3_lr_0.0005")
parser.add_argument('--epoch', type=int, default=89)
args = parser.parse_args()

molecule = args.mol
dir = args.model_dir
epoch = args.epoch

def num_heavy_plot(gen_smiles, train_smiles, molecule, output_dir):
    gen_dir = os.path.join(os.path.dirname(__file__), f"../{molecule}/outputs/{dir}/")

    train_smiles = random.sample(train_smiles, 10000)
    if len(gen_smiles) > 10000:
        gen_smiles = random.sample(gen_smiles, 10000)

    qed_gen = []
    qed_train = []

    m1 = 0
    for mol in gen_smiles:
        if Chem.MolFromSmiles(mol) is not None:
            num = Chem.MolFromSmiles(mol).GetNumHeavyAtoms()
            if num > m1:
                m1 = num
            qed_gen.append(num)

    m2 = 0
    for mol in train_smiles:
        if Chem.MolFromSmiles(mol) is not None:
            num = Chem.MolFromSmiles(mol).GetNumHeavyAtoms()
            if num > m2:
                m2 = num
            qed_train.append(num)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.hist(qed_gen, bins=list(range(m1)))
    ax2.hist(qed_train, bins=list(range(m2)))

    ax1.set_title(f'Generated {molecule}')
    ax1.set_xlabel('Number of Heavy Atoms')
    ax1.set_ylabel('Frequency')

    ax2.set_title(f'Training {molecule}')
    ax2.set_xlabel('Number of Heavy Atoms')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(output_dir)











