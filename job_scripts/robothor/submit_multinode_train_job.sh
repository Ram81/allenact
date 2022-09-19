#!/bin/bash
#SBATCH --job-name=procthor
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 8
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=robby,chappie,sophon,heistotron,
#SBATCH --output=slurm_logs/procthor-%j.out
#SBATCH --error=slurm_logs/procthor-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate procthor
echo "Source conda environment done!"

export PYTHONPATH=/srv/flash1/rramrakhya6/fall_2022/allenact

source /srv/flash1/rramrakhya6/fall_2022/vulkansdk-1.2.198.1/setup-env.sh
echo "Source vulkan done!"

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR
export DEFAULT_PORT=8739

unset DISPLAY

config=$1

SEED=12345
SUB_PROCESSES=1
EXPERIMENT_OUTPUT_DIR="experiment_output/object_nav_robothor_ddppo"

set -x

echo $PATH
echo $LD_LIBRARY_PATH
echo $VK_LAYER_PATH

export CUDA_LAUNCH_BLOCKING=1

echo "In RoboTHOR ObjectNav DD-PPO"
srun python allenact/main.py \
object_nav_robo_thor_ddppo \
-b projects/tutorials \
-m $SUB_PROCESSES \
-o $EXPERIMENT_OUTPUT_DIR \
-s $SEED \
--config_kwargs '{"distributed_nodes": 1, "headless": true, "num_train_processes": 8}' \
--distributed_ip_and_port $MASTER_ADDR:$DEFAULT_PORT
