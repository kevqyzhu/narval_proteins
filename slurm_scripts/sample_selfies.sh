#!/bin/bash 

#SBATCH --account=rrg-aspuru                        
#SBATCH --time=0-10:00            # time (DD-HH:MM)
#SBATCH --output=error/%x-%j.txt
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
source ~/sample_mol/bin/activate
# virtualenv --no-download ~/sample_gdb13_env
# pip3 install --no-index --upgrade pip
# pip3 install tqdm --no-index
# pip3 install torch --no-index
# pip3 install numpy --no-index
# pip3 install networkx
# pip3 install selfies


# seq_length=70
# n_hidden=$1
# drop_prob=$2
# lr=$3

# epoch=$4
# chem=unique94qed

chem=$1
seq_length=$2
n_hidden=$3
drop_prob=$4
lr=$5
epoch=$6
batch_size=$7

cd ../${chem}

model_dir=${chem}_outputs_seq_${seq_length}_hidden_${n_hidden}_dp_${drop_prob}_lr_${lr}_num_${batch_size}

outputs_directory=outputs/${model_dir}
model_directory=models/${chem}_models_selfies_seq_${seq_length}_hidden_${n_hidden}_dp_${drop_prob}_lr_${lr}
mkdir $outputs_directory
python3 ../RNN/sampler_smiles.py --batch_size $batch_size --n_hidden $n_hidden --drop_prob $drop_prob --lr $lr --vocab ../../data/${chem}_selfies.txt --tokens ../../data/${chem}_tokens.txt --model ${model_directory}/rnn_${epoch}_epoch.net > $outputs_directory/outputs${epoch}.txt

cd ../processing
python3 metrics.py --mol $chem --model_dir ${model_dir} --epoch $epoch