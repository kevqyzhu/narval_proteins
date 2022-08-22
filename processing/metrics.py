import sys
import os
import argparse
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="selfies_RNN")
parser.add_argument('--mol', default="largestfragments")
parser.add_argument('--model_dir', default="largestfragments_outputs_seq_2123_hidden_700_dp_0.3_lr_0.0005")
parser.add_argument('--epoch', type=int, default=89)
args = parser.parse_args()

model = args.model
mol = args.mol
model_dir = args.model_dir
epoch = args.epoch

gen_dir = os.path.join(os.path.dirname(__file__), f"../{model}/{mol}/outputs/{model_dir}/")


def is_valid(smiles: str):
    """
    Verifies whether a SMILES string corresponds to a valid molecule.
    Args:
        smiles: SMILES string
    Returns:
        True if the SMILES strings corresponds to a valid, non-empty molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    return smiles != '' and mol is not None and mol.GetNumAtoms() > 0

def metrics(gen_smiles, train_smiles):
    # train_smiles is list of training smiles
    # gen_smiles are list of smiles generated from the model
    num_gen = len(gen_smiles)
    valid_smiles = [] # all selfies produce valid smiles
    for smi in gen_smiles:
        if is_valid(smi):
            valid_smiles.append(smi)
    num_valid = len(valid_smiles)

    cans = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True) for smi in valid_smiles]
    unique_smiles = set(cans)
    num_unique = len(unique_smiles)

    # remove any training smiles that are in the unique smiles
    # train_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True) for smi in train_smiles]
    unique_smiles.difference_update(train_smiles)
    num_novel = len(unique_smiles)
    sys.stdout = open(f"{gen_dir}" + f"outputs{epoch}_benchmark.txt", "w")

    print('# generated : {}'.format(num_gen))
    print('# valid : {} validity   {}'.format(num_valid, num_valid/num_gen))
    print('# unique: {} uniqueness {}'.format(num_unique, num_unique/num_valid))
    print('# novel : {} novelty    {}'.format(num_novel, num_novel/num_unique))

    sys.stdout.close()

    return unique_smiles # we return the novel smiles


with open(f"{gen_dir}" + f"outputs{epoch}.txt", "r") as f:
    gen_smiles = f.read().split('\n')[:-1]


data = os.path.join(os.path.dirname(__file__), f"../data/")


with open(f"{data}" + f"{mol}.txt", "r") as f:
    train_smiles = f.read().split('\n')[:-1]

novel = metrics(gen_smiles, train_smiles)

with open(f"{gen_dir}" + f"novel_outputs{epoch}.txt", "w") as f:
    for smiles in novel:
        f.write("%s\n" % smiles)

from num_heavy_calc import num_heavy_plot
# from sa_calc import sa_plot

num_heavy_plot(novel, train_smiles, mol, gen_dir+f'hist_num_heavy_{epoch}.png')
# ring_plot(novel, train_smiles, mol, gen_dir+f'hist_rings_{epoch}.png')

