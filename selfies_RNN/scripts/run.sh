#!/bin/bash 
#SBATCH --account=rrg-aspuru                       
#SBATCH --time=2-00:00            # time (DD-HH:MM)
#SBATCH --output=../allab60/error/%x-%j.txt 
#SBATCH --job-name=test_allab60_selfies
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
source ~/proteins_rnn/bin/activate

cd ../allab60
directory=models/allab60_models_selfies_seq_1739_numlayers_3_hidden_700_dp_0.3_lr_0.0003
mkdir $directory
python3 ../../RNN/test.py --n_layers 3 --load_epoch -1 --tokens ../../data/allab60_tokens_selfies.txt --encoded ../../data/allab60_encoded_selfies.pickle --save_dir $directory --seq_length 1739 --n_epoch 200 --n_hidden 700 --drop_prob 0.3 --lr 0.0003 --batch_size 32
        