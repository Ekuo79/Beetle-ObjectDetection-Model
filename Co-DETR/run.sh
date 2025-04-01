#!/bin/bash
#SBATCH --job-name=CoDETR
#SBATCH --output=/blue/hulcr/eric.kuo/Co-DETR/fold3.out
#SBATCH --error=/blue/hulcr/eric.kuo/Co-DETR/fold3.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eric.kuo@ufl.edu
#SBATCH --account=hulcr
#SBATCH --qos=hulcr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --gpus=a100:3
#SBATCH --time=72:00:00

module purge
module load conda

# Activate the conda environment
conda activate CoDETR11

# Change to the specific directory
cd /blue/hulcr/eric.kuo/Co-DETR

# Run it
sh tools/dist_train.sh projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py 3 fold3

