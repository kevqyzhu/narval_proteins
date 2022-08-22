#!/bin/bash

cd ../pdbs2

count=0

for i in `ls`;
do
    n=$(ls $i | wc -l)
    count=$((count + n))
    echo $n
done

echo $count
