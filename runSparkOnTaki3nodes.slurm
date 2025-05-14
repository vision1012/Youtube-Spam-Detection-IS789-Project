#!/bin/bash
#SBATCH --job-name=sparkJob
#SBATCH --partition=high_mem
#SBATCH --nodes=3
#SBATCH --exclusive
#SBATCH --constraint=hpcf2018
#SBTACH --mem=64000
#SBATCH --qos=normal
#SBATCH --output=spark-job-%j-%u-slurm.out
#SBATCH --error=spark-job-%j-%u-slurm.out
#SBATCH --time=4:00:00

#Step 0: load Python module and set python file and arguments for the Spark job
export PYTHONPATH=/umbc/rs/is789sp25/common/software/anaconda3/bin
export PYSPARK_PYTHON=$PYTHONPATH/python
export PYSPARK_DRIVER_PYTHON=$PYTHONPATH/python
SPARK=/umbc/rs/is789sp25/common/software/spark-3.5.0-bin-hadoop3/
MY_SPARK=$(pwd)
SPARK_PYTHON_FILE=CODE.py
#SPARK_PYTHON_ARGUMENTS=100

#Setp 1: Create configuration and log folders for the Spark job
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
echo "Current Time : $current_time"
MY_SPARK=$MY_SPARK/spark-job-$SLURM_JOB_ID-$SLURM_NNODES-nodes-$current_time
mkdir -p $MY_SPARK
MY_SPARK_LOG_PATH=$MY_SPARK/logs
cp -r $SPARK/conf $MY_SPARK/spark-conf
export SPARK_CONF_DIR=$MY_SPARK/spark-conf
mkdir -p $MY_SPARK_LOG_PATH
echo "SPARK_CONF_DIR: $SPARK_CONF_DIR"
echo "SPARK_CONF_DIR: $SPARK_CONF_DIR" > $MY_SPARK_LOG_PATH/$current_time-log.txt


#Step 2: Update slaves file at $SPARK/conf based on the nodes allocated from job scheduler.
cat /dev/null > $SPARK_CONF_DIR/slaves
master=$(echo $SLURMD_NODENAME)
echo "master:$master"
echo "master:$master" >> $MY_SPARK_LOG_PATH/$current_time-log.txt

nodes=$(echo $SLURM_NODELIST | cut -d'[' -f 2 | cut -d']' -f 1)
nodes=$(scontrol show hostnames $SLURM_NODELIST)
echo "nodes: $nodes"
for element in $nodes; do
  echo "element1:$element"
  if [ $element != "$master" ]; then
    echo $element >> $SPARK_CONF_DIR/slaves
  fi
done

echo "slaves: $(cat $SPARK_CONF_DIR/slaves)"
echo "slaves: $(cat $SPARK_CONF_DIR/slaves)" >> $MY_SPARK_LOG_PATH/$current_time-log.txt

echo $(egrep --color 'Mem|Cache|Swap' /proc/meminfo)
echo $(ulimit -a)



#Step 3: Start/deploy Spark on all nodes allocated
$SPARK/sbin/stop-all.sh
#sleep 5

ulimit -c unlimited
#$SPARK/sbin/start-master.sh
$SPARK/sbin/start-all.sh >> $MY_SPARK_LOG_PATH/$current_time-log.txt
sleep 5

host=$(hostname)


Step 4: Submit your Spark job and wait for its finish
(time $SPARK/bin/spark-submit --master spark://$master:7077 --driver-memory 60g --executor-memory 60g --num-executors 5 --executor-cores 6 $SPARK_PYTHON_FILE >> $MY_SPARK_LOG_PATH/$current_time-log.txt) 2>> $MY_SPARK_LOG_PATH/$current_time-time.txt

#Step 5: Stop Spark at the end
sleep 5
$SPARK/sbin/stop-all.sh >> $MY_SPARK_LOG_PATH/$current_time-log.txt
