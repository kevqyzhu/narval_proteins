#!/bin/bash 
salloc --time=0-05:00 --gres=gpu:1 --mem=100G --account=rrg-aspuru --mail-user=kevqyzhu@gmail.com --mail-type=BEGIN --mail-type=END --mail-type=FAIL --mail-type=REQUEUE

module load nixpkgs/16.09  
module load python/3.7.4
module load gcc/7.3.0
module load openbabel/2.4.1 
module load networkx
module load rdkit/2019.03.4
source ~/proteins_rnn/bin/activate