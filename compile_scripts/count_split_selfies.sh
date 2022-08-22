#!/bin/bash 
#SBATCH --account=rrg-aspuru                  
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-12:00            # time (DD-HH:MM)
#SBATCH --output=%x-%j.txt
#SBATCH --nodes=1
#SBATCH --mail-user=kevqyzhu@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE 
 
module load CCEnv
module load StdEnv
module load nixpkgs/16.09  
module load python/3.7
module load gcc/7.3.0
module load intel/2018.3
module load openbabel/2.4.1 
module load networkx
# virtualenv --no-download ~/count_selfies
source ~/count_selfies/bin/activate
# pip3 install selfies
# pip3 install matplotlib

cd ../python_scripts
python3 selfies_split.py