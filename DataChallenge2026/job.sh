#!/bin/bash
#SBATCH --job-name=version1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /home/infres/alaroye-25/data_challenge

# Environnement propre
module purge

# Charger les modules nécessaires
module load python/3.12
module load cuda/12.4

export LD_LIBRARY_PATH=/projects/share/apps/miniconda3/25.5.1/lib:$LD_LIBRARY_PATH

source venv/bin/activate

nvidia-smi

python DataChallenge2026/main.py