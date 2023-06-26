#!/bin/bash
# usage : change --nodes and WORLD_SIZE and run : sbatch run_train.sh
#SBATCH --job-name=xn_train
#SBATCH --partition=gpu
#SBATCH --time=17280
#SBATCH --wait-all-nodes=1
### e.g. request 2 nodes with 4 gpu each, totally 8 gpus (WORLD_SIZE==8)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=4
#SBATCH --error=jobs/job.%J.err
#SBATCH --output=jobs/job.%J.out
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12355
export WORLD_SIZE=4

### get the first node name as master address
echo "NODELIST="${SLURM_NODELIST} 
export MASTER_ADDR=$(getent hosts $(hostname) | awk '{ print $1 }')
echo $(hostname)
echo "MASTER ADDRESS="${MASTER_ADDR}

# activate venv
source /home/groups/graylab_share/OMERO.rdsStore/paganol/.bashrc
conda activate test

# srun commands
srun python3 ${1} &
echo "ran, waiting for finish"
wait
rm ${1}

