#!/bin/bash 
#SBATCH --account=rrg-aspuru                       
#SBATCH --time=2-00:00            # time (DD-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=kevqyzhu@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE 

chem=$1
seq_length=$2
n_hidden=$3
drop_prob=$4
lr=$5
load_epoch=$6
n_epoch=$7
batch_size=$8

module load nixpkgs/16.09  
module load python/3.7
module load gcc/8.3.0
module load networkx
virtualenv --no-download ~/train_guacamol_rnn
source ~/train_guacamol_rnn/bin/activate
# curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py
# python get-pip.py pip==19.3.1
# rm get-pip.py
# pip3 install --no-index --upgrade pip
# pip3 install torch --no-index
# pip3 install numpy --no-index
# pip3 install networkx
# pip3 install selfies

cd ../${chem}
directory=models/${chem}_models_selfies_seq_${seq_length}_hidden_${n_hidden}_dp_${drop_prob}_lr_${lr}
mkdir $directory
python3 ../RNN/test_selfies.py --load_epoch $load_epoch --vocab ../../data/${chem}_selfies.txt --tokens ../../data/${chem}_tokens.txt --save_dir $directory --seq_length $seq_length --n_epoch $n_epoch --n_hidden $n_hidden --drop_prob $drop_prob --lr $lr --batch_size $batch_size 


# > ${chem}_models_selfies_seq_${seq_length}_hidden_${n_hidden}_dp_${drop_prob}_lr_${lr}.txt

# for i in `seq 75 25 200`; ### Inner for loop ###
#     do
#         sbatch guac_dist_bench.sh $n_hidden $drop_prob $lr $i
#         sbatch guac_directed_bench.sh $n_hidden $drop_prob $lr $i
#     done