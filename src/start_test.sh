######################Installed packages#################################
# File: start_test.sh
# Description: 
#	Downloads the sql DB and loads in the mysql. 
#	Starts the Assignment, load the tables and perform unit test
# 	Packages installed on my system:
#		apt-get install -y mariadb-server
# 		apt-get install -y libmariadb-dev
# 		apt-get install -y libmariadb-dev-compat
#########################################################################

#!/bin/bash

BASEBALL_TAR_FILE=baseball.sql.tar.gz
BASEBALL_SQL_BASEBALL_TAR_FILE=baseball.sql
PW=$1

function cleanup {
	killall -9 mysql 2>/dev/null
}

if [ -z "$1" ]; then
	echo "Root Password not given. Usage: ./start_test.sh <root pw>"
        exit 1;
fi

if [[ $EUID -ne 0 ]]; then
        echo "This script must be run as root"
#        exit 1;
fi

echo "Deleting old database"
mysql -u root -p${PW} -e "DROP DATABASE network"

echo "Creating database"
mysql -u root -p${PW} -e "CREATE DATABASE network"

mysql -u root -p${PW} -e "use network"

# Create table
csvsql --dialect mysql --snifflimit 1000000 ../data/dataset.csv > ../data/data.sql

mysql -u root -p${PW} network < ../data/data.sql

echo "Loading database. This may take a while..."
mysql -u root -p${PW} -e "use network; load data local infile '../data/dataset.csv' into table dataset fields terminated by ',' ignore 1 rows "

#python3 assignment.py

echo "Test complete. Cleanup repository"
#trap cleanup EXIT
