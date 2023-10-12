#!/bin/sh

rm -rf plots
mkdir -p results
mkdir -p plots/importance 
mkdir -p plots/after_trimming
mkdir -p plots/before_trimming

echo "##########################kfold-random##################################"
python3 main.py ../data/dataset.csv class3 10 kfold random quick >> results/kfold_random
#echo "##########################kfold-half-gradient##################################"
#python3 main.py ../data/dataset.csv class3 10 kfold half-gradient quick >> results/kfold_half-gradient
#echo "##########################kfold-gradient##################################"
#python3 main.py ../data/dataset.csv class3 10 kfold gradient quick >> results/kfold_gradient

