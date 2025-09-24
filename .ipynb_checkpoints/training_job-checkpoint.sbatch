#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=conv5+
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=unet_%j.txt
#SBATCH --error=unet_%j.txt

# Set container image path
export SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/develop.sif

# Run your training script inside the container
singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch \
    $SINGULARITY_IMAGE_PATH python run_training.py