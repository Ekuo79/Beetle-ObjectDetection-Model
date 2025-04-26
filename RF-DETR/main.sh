#!/bin/bash
#SBATCH --job-name=rfdetr
#SBATCH --output=/blue/hulcr/eric.kuo/rfdetr/run3.out
#SBATCH --error=/blue/hulcr/eric.kuo/rfdetr/run3.err
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

conda activate rfdetr

# Change to the specific directory
cd /blue/hulcr/eric.kuo/rfdetr


python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py

