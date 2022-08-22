chem=largesmiles

. ../${chem}/sample_params.sh

for i in `seq $n_hidden_min $n_hidden_incr $n_hidden_max`;      
do
    for j in `seq $drop_prob_min $drop_prob_incr $drop_prob_max`; 
    do
          for k in `seq $lr_min $lr_incr $lr_max`; 
            do  
                if [ -d "../${chem}/models/${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k}" ]
                then
                    epoch=$(ls ../${chem}/models/${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k} | sed -e s/[^0-9]//g |sort -n|tail -n 1)
                    echo $epoch
                    sbatch --job-name=sample_${chem}_selfies --output=../${chem}/error/%x-%j.txt sample_selfies.sh $chem $seq_length $i $j $k $epoch $batch_size
                else
                    echo "Error: Directory ${chem}_models_selfies_seq_${seq_length}_hidden_${i}_dp_${j}_lr_${k} does not exists."
                fi
            done
    done

done
