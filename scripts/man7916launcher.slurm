#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:15:00
#SBATCH --error=results-%J.err
#SBATCH --output=results-%J.out
#SBATCH --job-name=MAN7916Setup
#SBATCH --gres=gpu:1

# load modules
module load cuda/cuda-12.6.0
module load python/python-3.11.4-oneapi-2023.1.0

# Get git repo with sparse checkout
cd ~/MAN7916
git init
git remote add origin https://github.com/amckenny/MAN7916.git
git sparse-checkout init --no-cone
git sparse-checkout set scripts corpora
git fetch --depth=1 origin main
git checkout -f FETCH_HEAD

# Set up virtual environment and install packages
source ~/MAN7916/.venv/bin/activate
python3 -m spacy download en_core_web_sm
python3 ./nltk_environ_setup.py
