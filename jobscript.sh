#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=new_model
#SBATCH --output=logs/max_pool
#SBATCH --error=error/max_pool
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:16G

/cluster/scratch/alakshmanan/miniconda3/envs/occupancy/bin/python /cluster/home/alakshmanan/sem_proj/nerfacc2/occupanynerf/examples/train_ngp_nerf.py

