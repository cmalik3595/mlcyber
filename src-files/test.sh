#!/bin/sh

rm -rf plots
mkdir -p results
mkdir -p plots/importance 
mkdir -p plots/after_trimming
mkdir -p plots/before_trimming

echo "##########################kfold-random##################################"
python3 main.py ../data/dataset.csv class3 10 kfold random quick >> results/kfold_random
echo "##########################kfold-half-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 kfold half-gradient quick >> results/kfold_half-gradient
echo "##########################kfold-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 kfold gradient quick >> results/kfold_gradient

echo "######################################################################################"
echo "######################################################################################"

echo "##########################tss-random##################################"
python3 main.py ../data/dataset.csv class3 10 tss random quick >> results/tss_random
echo "##########################tss-half-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 tss half-gradient quick >> results/tss_half-gradient
echo "##########################tss-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 tss gradient quick >> results/tss_gradient

echo "######################################################################################"
echo "######################################################################################"
echo "##########################rkfold-random##################################"

python3 main.py ../data/dataset.csv class3 10 rkfold random quick >> results/rkfold_random
echo "##########################rkfold-half-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 rkfold half-gradient quick >> results/rkfold_half-gradient
echo "##########################rkfold-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 rkfold gradient quick >> results/rkfold_gradient

echo "######################################################################################"
echo "######################################################################################"
