import itertools
import os
import sys
from math import log10, floor
from grid_search import round_sig
import re
import time

def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def sample(chem):
    params, grid_params = get_sample_params(chem)
    keys, values = zip(*grid_params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]


    # permutations_dicts[0].update(params)
    # sample_latest(chem, permutations_dicts[0])

    for d in permutations_dicts:
        d.update(params)
        sample_latest(chem, d)
    
def get_sample_params(chem):
    if chem == "all-fragments":
        params = {"seq_length": 24833,
                "batch_size": 10000}

        params_grid = {"drop_prob": [round_sig(x * 0.1, 1) for x in range(1, 4)],
                      "lr": [round_sig(x * 0.0001, 1) for x in range(1, 4)],
                      "n_hidden": list(range(1500, 2000, 200)),
                      "n_layers": list(range(2, 4, 1))}

        return (params, params_grid)
    
    elif chem == "fragments_gr_1000":
        params = {"seq_length": 14252, #22786
                "batch_size": 10000}

        params_grid = {"drop_prob": [round_sig(x * 0.1, 1) for x in range(1, 4)],
                      "lr": [round_sig(x * 0.0001, 1) for x in range(1, 4)],
                      "n_hidden": list(range(1500, 2000, 200)),
                      "n_layers": list(range(2, 4, 1))}

        return (params, params_grid)

    elif chem == "fragments_500_20000":
        params = {"seq_length": 14252, #22786
                "batch_size": 10000}

        params_grid = {"drop_prob": [round_sig(x * 0.1, 1) for x in range(1, 4)],
                      "lr": [round_sig(x * 0.0001, 1) for x in range(1, 4)],
                      "n_hidden": list(range(1500, 2000, 200)),
                      "n_layers": list(range(2, 4, 1))}

        return (params, params_grid)

    elif chem == "all-proteins":
        params = {"seq_length": 1000,
                "batch_size": 10000}

        params_grid = {"drop_prob": [round_sig(x * 0.1, 1) for x in range(1, 4)],
                      "lr": [round_sig(x * 0.0001, 1) for x in range(1, 4)],
                      "n_hidden": list(range(1500, 2000, 100)),
                      "n_layers": list(range(2, 4, 1))}

        return (params, params_grid)
        
    elif chem == "oled":
        params = {"seq_length": 251,
                "batch_size": 10000}

        params_grid = {"drop_prob": [round_sig(x * 0.1, 1) for x in range(1, 4)],
                      "lr": [round_sig(x * 0.0001, 1) for x in range(1, 4)],
                      "n_hidden": list(range(400, 700, 100)),
                      "n_layers": list(range(2, 4, 1))}

        return (params, params_grid)

    elif chem == "allab60":
        params = {"seq_length": 1739,
                "batch_size": 10000}

        params_grid = {"drop_prob": [round_sig(x * 0.1, 1) for x in range(1, 4)],
                      "lr": [round_sig(x * 0.0001, 1) for x in range(1, 4)],
                      "n_hidden": [350, 700],
                      "n_layers": [1, 3]}

        return (params, params_grid)


def sample_latest(chem, params):
    model_directory=f"../{chem}/models/{chem}_models_selfies_seq_{params['seq_length']}_numlayers_{params['n_layers']}_hidden_{params['n_hidden']}_dp_{params['drop_prob']}_lr_{params['lr']}"
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


            # params['epoch'] = epoch_nums[len(epoch_nums)//2 -1]
            # print(params['epoch'])
            # print("\n")
    

def write_sh_script(chem, params):
    with open (f'run_sample.sh', 'w') as rsh:
        rsh.write(f'''\
#!/bin/bash 

#SBATCH --account=rrg-aspuru                        
#SBATCH --time=0-10:00            # time (DD-HH:MM)
#SBATCH --output=../{chem}/error/%x-%j.txt 
#SBATCH --job-name=sample_{chem}_selfies
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-user=kevqyzhu@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE 

module load nixpkgs/16.09  
module load python/3.7
module load gcc/8.3.0
module load networkx
module load rdkit/2019.03.4
source ~/proteins_rnn/bin/activate

cd ../{chem}

dir={chem}_outputs_selfies_seq_{params['seq_length']}_numlayers_{params['n_layers']}_hidden_{params['n_hidden']}_dp_{params['drop_prob']}_lr_{params['lr']}_num_{params['batch_size']}
outputs_directory=outputs/${{dir}}
mkdir $outputs_directory

model_directory=models/{chem}_models_selfies_seq_{params['seq_length']}_numlayers_{params['n_layers']}_hidden_{params['n_hidden']}_dp_{params['drop_prob']}_lr_{params['lr']}
python3 ../../RNN/RNN_sampler.py --n_layers {params['n_layers']} --model_type selfies --batch_size {params['batch_size']} --n_hidden {params['n_hidden']} --drop_prob {params['drop_prob']} --lr {params['lr']} --tokens ../../data/{chem}_tokens_selfies.txt --model ${{model_directory}}/rnn_{params["epoch"]}_epoch.net --output_file $outputs_directory/outputs{params["epoch"]}.txt

cd ../../processing
python3 metrics.py --model selfies_RNN --mol {chem} --model_dir $dir --epoch {params["epoch"]}
        ''')

    os.system("chmod +x run_sample.sh")

    # os.system("./run_sample.sh")
    os.system("sbatch run_sample.sh")

    # os.remove("run.sh")


if __name__ == "__main__":
    sys.setrecursionlimit(4000)
    chem = "allab60"
    sample(chem)