######################Installed packages#################################
# File: start_test.sh
# Description: 
#	Downloads the sql DB and loads in the mysql. 
#########################################################################

#!/bin/bash

python3 Traditional-ML-Training.py | tee traditional.txt
python3 DL-Training-updated.py | Traditional-NN.txt

python3 main.py ../data/dataset.csv class3 10 kfold random quick | tee tuneup.txt
