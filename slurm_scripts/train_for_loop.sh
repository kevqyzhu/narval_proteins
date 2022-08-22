chem=hmd

. ../${chem}/train_params.sh


for i in `seq $n_hidden_min $n_hidden_incr $n_hidden_max`;      
do
    for j in `seq $drop_prob_min $drop_prob_incr $drop_prob_max`; 
    do
          for k in `seq $lr_min $lr_incr $lr_max`; 
                do
                    sbatch --job-name=test_${chem}_selfies --output=../${chem}/error/%x-%j.txt test_selfies.sh $chem $seq_length $i $j $k -1 $num_epoch $batch_size
                done
    done

done
