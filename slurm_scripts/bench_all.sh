#!/bin/bash 

chem=guacamol_train
seq_length=109

n_hidden_min=400
n_hidden_incr=100 
n_hidden_max=600

drop_prob_min=0.1
drop_prob_incr=0.1
drop_prob_max=0.3

lr_min=0.0001
lr_incr=0.0001
lr_max=0.0003

cd ../${chem}/models


for i in `seq $n_hidden_min $n_hidden_incr $n_hidden_max`;      
do
    for j in `seq $drop_prob_min $drop_prob_incr $drop_prob_max`; 
    do
          for k in `seq $lr_min $lr_incr $lr_max`; 
                do
                    if [ -d "${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k}" ] 
                    then
                        # echo "Directory ${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k} exists." 
                        cd ${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k}
                        epoch=$(ls -d -- *.net | sed -e s/[^0-9]//g |sort -n|tail -n 1)
                        cd ..
                        echo $epoch
                        if [[ $epoch -eq 200 ]]
                        then
                            for l in `seq 150 25 200`; ### Inner for loop ###
                            do
                                cd ..
                                # sbatch guac_dist_bench.sh $i $j $k $l
                                sbatch guac_directed_bench.sh $i $j $k $l
                                cd models
                            done
                        else
                            cd ..
                            # sbatch guac_dist_bench.sh $i $j $k $l
                            sbatch guac_directed_bench.sh $i $j $k $epoch
                            cd models
                        fi
                    else
                        echo "Error: Directory ${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k} does not exists."
                        cd ..
                        sbatch test_${chem}.sh $i $j $k -1
                        cd models
                    fi
                done
    done

done

