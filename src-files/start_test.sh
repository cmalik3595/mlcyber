######################Installed packages#################################
# File: start_test.sh
# Description: 
#	Downloads the sql DB and loads in the mysql. 
#########################################################################

python3 -u Traditional-ML-Training.py | tee traditional.txt
python3 -u DL-Training-updated.py | tee traditional-NN.txt

python3 -u main.py ../data/dataset.csv class3 10 kfold random quick | tee tuneup.txt
