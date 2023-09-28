#!/bin/sh

rm -rf plots
mkdir -p plots/importance 
mkdir -p plots/after_trimming
mkdir -p plots/before_trimming

python3 main.py ../data/dataset.csv class3 10 kfold random quick
