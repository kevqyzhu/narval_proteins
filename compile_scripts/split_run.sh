#!/bin/bash 
#SBATCH --account=rrg-aspuru                  
#SBATCH --ntasks-per-node=8
#SBATCH --array=1-20
#SBATCH --time=0-0:50            # time (DD-HH:MM)
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
# virtualenv --no-download ~/load_proteins
source ~/load_proteins/bin/activate
# pip3 install pypdb

num_split=20

# cd ../python_scripts
# python3 get_pdbs.py --num_split $num_split --split $SLURM_ARRAY_TASK_ID
# cd ../pdbs
# for j in `ls ${SLURM_ARRAY_TASK_ID}`;      
# do
#     obabel ${SLURM_ARRAY_TASK_ID}/${j} -ocan -O ../smiles/allmols/${j}.txt
# done


for i in `seq 1 1 ${num_split}`;      
do
    cd ../python_scripts
    python3 get_pdbs.py --num_split $num_split --split $i &
    # cd ../pdbs
    # for j in `ls $i`;      
    # do
    #     obabel ${i}/${j} -ocan -O ../smiles/allmols/${j}.txt
    # done

done


# python ../python_scripts/get_pdbs.py
# cd ../smiles
# ./convert_pdb_smi.sh