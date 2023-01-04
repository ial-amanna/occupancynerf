#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=new_model
#SBATCH --output=logs/logs11
#SBATCH --error=error/error11
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:16G

/cluster/scratch/alakshmanan/miniconda3/envs/nerfacc/bin/python /cluster/home/alakshmanan/sem_proj/nerfacc2/occupanynerf/examples/train_ngp_nerf.py

