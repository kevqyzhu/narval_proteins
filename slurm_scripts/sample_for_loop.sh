chem=hmd

. ../${chem}/sample_params.sh


for i in `seq $n_hidden_min $n_hidden_incr $n_hidden_max`;      
do
    for j in `seq $drop_prob_min $drop_prob_incr $drop_prob_max`; 
    do
          for k in `seq $lr_min $lr_incr $lr_max`; 
            do
                for l in `seq 100 25 100`
                do 
                    sbatch --job-name=sample_${chem}_selfies --output=../${chem}/error/%x-%j.txt sample_selfies.sh $chem $seq_length $i $j $k $l $batch_size
                done
            done
    done

done
