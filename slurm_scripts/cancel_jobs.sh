#!/bin/bash 

job_type=test_allab60_smiles 

for i in `sq -u kevqyzhu | grep "${job_type}" | awk '{print $1}'`;
do 
scancel $i
done