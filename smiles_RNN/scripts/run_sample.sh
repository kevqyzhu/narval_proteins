#!/bin/bash 

#SBATCH --account=rrg-aspuru                        
#SBATCH --time=0-10:00            # time (DD-HH:MM)
#SBATCH --output=../allab60/error/%x-%j.txt 
#SBATCH --job-name=sample_allab60_smiles
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

cd ../allab60

dir=allab60_outputs_smiles_seq_3461_numlayers_3_hidden_700_dp_0.3_lr_0.0003_num_10000
outputs_directory=outputs/${dir}
mkdir $outputs_directory

model_directory=models/allab60_models_smiles_seq_3461_numlayers_3_hidden_700_dp_0.3_lr_0.0003
python3 ../../RNN/RNN_sampler.py --n_layers 3 --model_type smiles --batch_size 10000 --n_hidden 700 --drop_prob 0.3 --lr 0.0003 --tokens ../../data/allab60_tokens_smiles.txt --model ${model_directory}/rnn_182_epoch.net --output_file $outputs_directory/outputs182.txt

cd ../../processing
python3 metrics.py --model smiles_RNN --mol allab60 --model_dir $dir --epoch 182
        