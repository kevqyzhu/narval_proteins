import itertools
import os
import sys
from math import log10, floor
import time

def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def grid_search(chem):
    params, grid_params = get_train_params(chem)
    keys, values = zip(*grid_params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # permutations_dicts[0].update(params)
    # write_sh_script(chem, permutations_dicts[0])
    for d in permutations_dicts:
        d.update(params)
        write_sh_script(chem, d)

def gird_search_latest(chem):
    params, grid_params = get_train_params(chem)
    keys, values = zip(*grid_params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # permutations_dicts[0].update(params)
    # sample_latest(chem, permutations_dicts[0])

    for d in permutations_dicts:
        d.update(params)
        train_latest(chem, d)

def train_latest(chem, params):
    model_directory=f"../{chem}/models/{chem}_models_smiles_seq_{params['seq_length']}_numlayers_{params['n_layers']}_hidden_{params['n_hidden']}_dp_{params['drop_prob']}_lr_{params['lr']}"
    if os.path.isdir(model_directory):
        epochs = os.listdir(model_directory)

        # for epoch in epochs:
            # path = os.path.join(model_directory, epoch)
            # if os.stat(path).st_mtime < time.time() - 437960:
            #     print(path)
            #     os.remove(path)

        epoch_nums = [int("".join(filter(str.isdigit, x))) for x in epochs 
        if "".join(filter(str.isdigit, x)) != '']
        epoch_nums.sort()
        if len(epoch_nums) >= 1:
            params['epoch'] = epoch_nums[-1]
            print(params['epoch'])
            write_sh_script(chem, params)
    
def get_train_params(chem):
    if chem == "all-fragments":
        params = {"seq_length": 31996,
                "load_epoch": -1,
                "n_epoch": 200,
                "batch_size": 5}

        params_grid = {"drop_prob": [round_sig(x * 0.1, 1) for x in range(1, 4)],
                      "lr": [round_sig(x * 0.0001, 1) for x in range(1, 4)],
                      "n_hidden": list(range(1500, 2000, 100)),
                      "n_layers": list(range(2, 4, 1))}

        return (params, params_grid)

    elif chem == "all-proteins":
        params = {"seq_length": 1000,
                "load_epoch": -1,
                "n_epoch": 200,
                "batch_size": 10}

        params_grid = {"drop_prob": [round_sig(x * 0.1, 1) for x in range(1, 4)],
                      "lr": [round_sig(x * 0.0001, 1) for x in range(1, 4)],
                      "n_hidden": list(range(1500, 2000, 100)),
                      "n_layers": list(range(2, 4, 1))}

        return (params, params_grid)

    elif chem == "oled":
        params = {"seq_length": 251,
                "load_epoch": -1,
                "n_epoch": 200,
                "batch_size": 32}

        params_grid = {"drop_prob": [round_sig(x * 0.1, 1) for x in range(1, 4)],
                      "lr": [round_sig(x * 0.0001, 1) for x in range(1, 4)],
                      "n_hidden": list(range(400, 700, 100)),
                      "n_layers": list(range(2, 4, 1))}

        return (params, params_grid)

    elif chem == "allab60":
        params = {"seq_length": 3461,
                "load_epoch": -1,
                "n_epoch": 200,
                "batch_size": 32}

        params_grid = {"drop_prob": [round_sig(x * 0.1, 1) for x in range(1, 4)],
                      "lr": [round_sig(x * 0.0001, 1) for x in range(1, 4)],
                      "n_hidden": [500, 700],
                      "n_layers": [1, 3]}

        return (params, params_grid)

def write_sh_script(chem, params):
    with open (f'run.sh', 'w') as rsh:
        rsh.write(f'''\
#!/bin/bash 
#SBATCH --account=rrg-aspuru                       
#SBATCH --time=2-00:00            # time (DD-HH:MM)
#SBATCH --output=../{chem}/error/%x-%j.txt 
#SBATCH --job-name=test_{chem}_smiles
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-user=kevqyzhu@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE 

module load nixpkgs/16.09  
module load python/3.7
module load gcc/8.3.0
module load networkx
source ~/proteins_rnn/bin/activate

cd ../{chem}
directory=models/{chem}_models_smiles_seq_{params['seq_length']}_numlayers_{params['n_layers']}_hidden_{params['n_hidden']}_dp_{params['drop_prob']}_lr_{params['lr']}
mkdir $directory
python3 ../../RNN/test.py --n_layers {params['n_layers']} --load_epoch {params['load_epoch']} --tokens ../../data/{chem}_tokens_smiles.txt --encoded ../../data/{chem}_encoded_smiles.pickle --save_dir $directory --seq_length {params['seq_length']} --n_epoch {params['n_epoch']} --n_hidden {params['n_hidden']} --drop_prob {params['drop_prob']} --lr {params['lr']} --batch_size {params['batch_size']}
        ''')
    os.system("chmod +x run.sh")
    # os.system("./run.sh")
    os.system("sbatch run.sh")
    # os.remove("run.sh")


if __name__ == "__main__":
    sys.setrecursionlimit(4000)
    chem = "allab60"
    gird_search_latest(chem)