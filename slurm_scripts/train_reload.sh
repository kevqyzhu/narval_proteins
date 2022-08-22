chem=guacamol_train

. ../${chem}/train_params.sh


for i in `seq $n_hidden_min $n_hidden_incr $n_hidden_max`;      
do
    for j in `seq $drop_prob_min $drop_prob_incr $drop_prob_max`; 
    do
          for k in `seq $lr_min $lr_incr $lr_max`; 
                do
                    if [ -d "../${chem}/models/${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k}" ] 
                    then
                        # echo "Directory ${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k} exists." 
                        epoch=$(ls ../${chem}/models/${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k} | sed -e s/[^0-9]//g |sort -n|tail -n 1)
                        echo $epoch
                        if [ $epoch -lt $num_epoch ]
                        then
                            sbatch --job-name=test_${chem}_selfies --output=../${chem}/error/%x-%j.txt test_selfies.sh $chem $seq_length $i $j $k $epoch $num_epoch $batch_size
                        fi
                    else
                        echo "Error: Directory ${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k} does not exists."
                        sbatch --job-name=test_${chem}_selfies --output=../${chem}/error/%x-%j.txt test_selfies.sh $chem $seq_length $i $j $k -1 $num_epoch $batch_size
                    fi
                done
    done

done
