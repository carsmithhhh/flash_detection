#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=flashdet_train
#SBATCH --gpus 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=train_%j.txt
#SBATCH --error=train_%j.txt

# Container image with the Python/PyTorch environment.
export SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/develop.sif

# Pick the model by swapping the config file.
CONFIG=${1:-configs/conformer.yaml}

# Turn "verbose=False" in the tqdm progress bar in flashdet/engine.py!!!
singularity exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch \
    $SINGULARITY_IMAGE_PATH python train.py --config "$CONFIG"
