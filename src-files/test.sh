#!/bin/sh

rm -rf plots
mkdir -p plots/importance 
mkdir -p plots/after_trimming
mkdir -p plots/before_trimming

#echo "##########################kfold-random##################################"
#python3 main.py ../data/dataset.csv class3 10 kfold random quick
echo "##########################kfold-half-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 kfold half-gradient quick
echo "##########################kfold-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 kfold gradient quick

echo "##########################tss-random##################################"
python3 main.py ../data/dataset.csv class3 10 tss random quick
echo "##########################tss-half-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 tss half-gradient quick
echo "##########################tss-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 tss gradient quick

echo "##########################rkfold-random##################################"
python3 main.py ../data/dataset.csv class3 10 rkfold random quick
echo "##########################rkfold-half-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 rkfold half-gradient quick
echo "##########################rkfold-gradient##################################"
python3 main.py ../data/dataset.csv class3 10 rkfold gradient quick
