#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=new_model
#SBATCH --output=logs/final_check
#SBATCH --error=error/final_check
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:16G


/cluster/scratch/alakshmanan/miniconda3/envs/occupancy/bin/python /cluster/home/alakshmanan/sem_proj/nerfacc2/occupanynerf/train_instant_ngp.py --radius_encoded=0.1
# /cluster/scratch/alakshmanan/miniconda3/envs/occupancy/bin/python /cluster/home/alakshmanan/sem_proj/nerfacc2/occupanynerf/get_esdf.py
# /cluster/scratch/alakshmanan/miniconda3/envs/occupancy/bin/python /cluster/home/alakshmanan/sem_proj/nerfacc2/occupanynerf/get_metrics.py
