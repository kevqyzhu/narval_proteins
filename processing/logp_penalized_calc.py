import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import time, os
import numpy as np
import multiprocessing
import random
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from sascorer import calculateScore
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mol', default="logp")
parser.add_argument('--model_dir', default="logp_outputs_seq_87_hidden_400_dp_0.3_lr_0.0001")
parser.add_argument('--epoch', type=int, default=150)
args = parser.parse_args()

molecule = args.mol
dir = args.model_dir
epoch = args.epoch
data = os.path.join(os.path.dirname(__file__), f"../../data/")

logP_values = np.loadtxt(data + 'logp_values.txt')
SA_scores = np.loadtxt(data + 'SA_scores.txt')
cycle_scores = np.loadtxt(data + 'cycle_scores.txt')
SA_mean =  np.mean(SA_scores)
SA_std=np.std(SA_scores)
logP_mean = np.mean(logP_values)
logP_std= np.std(logP_values)
cycle_mean = np.mean(cycle_scores)
cycle_std=np.std(cycle_scores)


def log_print(fname, text):
    with open(fname, 'a') as f:
        f.write(text + '\n')


def logp_score(smiles):

    m = Chem.MolFromSmiles(smiles)
    try:
        logp = MolLogP(m)
    except:
        return 0

    SA_score = -calculateScore(m)
    cycle_list = m.GetRingInfo().AtomRings()
    if len(cycle_list) == 0:
      cycle_length = 0
    else:
      cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
      cycle_length = 0
    else:
      cycle_length = cycle_length - 6

    cycle_score = -cycle_length
    SA_score_norm=(SA_score-SA_mean)/SA_std
    logp_norm=(logp-logP_mean)/logP_std
    cycle_score_norm=(cycle_score-cycle_mean)/cycle_std
    score_one = SA_score_norm + logp_norm + cycle_score_norm
    return score_one


def chunks(lst, n=1000000):
    """Yield successive n-sized chunks from lst."""
    size = len(lst)
    sizes = list(range(0, size, n))
    chunks = [lst[i:i + n] for i in sizes]
    return chunks


def logp_penalized_plot(molecule, dir, epoch):
    gen_dir = os.path.join(os.path.dirname(__file__), f"../{molecule}/outputs/{dir}/")

    with open(f"{gen_dir}" + f"novel_outputs{epoch}.txt", "r") as f:
        gen_smiles = f.read().split('\n')[:-1]

    data = os.path.join(os.path.dirname(__file__), f"../../data/")

    with open(f"{data}" + f"{molecule}.txt", "r") as f:
        train_smiles = f.read().split('\n')[:-1]

    train_smiles = random.sample(train_smiles, 10000)
    if len(gen_smiles) > 10000:
        gen_smiles = random.sample(gen_smiles, 10000)

    qed_gen = []
    qed_train = []

    m1 = 0
    for mol in gen_smiles:
        if Chem.MolFromSmiles(mol) is not None:
            try:
                num = logp_score(mol)
            except:
                num = 0
            if num > m1:
                m1 = num
            qed_gen.append(num)

    m2 = 0
    for mol in train_smiles:
        if Chem.MolFromSmiles(mol) is not None:
            try:
                num = logp_score(mol)
            except:
                num = 0
            if num > m2:
                m2 = num
            qed_train.append(num)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    majorLocator = MultipleLocator(0.1)
    majorFormatter = FormatStrFormatter('%.1f')
    minorLocator = MultipleLocator(0.01)

    ax1.hist(qed_gen, bins=[x * 0.1 for x in range(0, 170)])
    ax2.hist(qed_train, bins=[x * 0.1 for x in range(0, 170)])

    ax1.set_title(f'Generated {molecule}')
    ax1.set_xlabel('LogP Score')
    ax1.set_ylabel('Frequency')

    ax2.set_title(f'Training {molecule}')
    ax2.set_xlabel('LogP Score')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(gen_dir+f'hist_logp_{epoch}.png')


if __name__ == '__main__':

    logp_penalized_plot(molecule, dir, epoch)

    # file = data + 'logp.txt'
    # s = open(file,'r')
    # smiles = [line.strip().split(" ")[0] for line in s.readlines() if line != '\n']
    # smiles_lists = chunks(smiles, 1000000)
    #
    # for smiles in smiles_lists:
    #     start=time.time()
    #     with multiprocessing.Pool() as pool: # max cpu
    #         scores = pool.map(logp_score, smiles)
    #     smiles = np.array(smiles)
    #     scores = np.array(scores)
    #     #idx = np.argsort(scores)[-1000:]
    #     log_print('logs',"scores mean {} max {} ".format(np.mean(scores), max(scores)))
    #     smiles = smiles[scores>4].tolist()
    #     n = len(smiles)
    #     if n > 0 :
    #         smiles = "\n".join(smiles)
    #         scores = scores[scores>4].tolist()
    #         scores_ = "\n".join([str(s) for s in scores])
    #         log_print('scsmiles.txt', smiles)
    #         log_print('scs.txt', scores_)
    #     end = time.time()
    #     total_time = (end-start)/60
    #     log_print('logs',"minutes {} scores {} ".format(total_time, n))
