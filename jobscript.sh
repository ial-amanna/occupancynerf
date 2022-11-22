#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=debug5
#SBATCH --output=logs4
#SBATCH --error=error4
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:16G

/cluster/scratch/alakshmanan/miniconda3/envs/ingp/bin/python /cluster/home/alakshmanan/sem_proj/nerfacc/examples/train_ngp_nerf.py --train_split train --scene lego

