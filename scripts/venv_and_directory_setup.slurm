#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:15:00
#SBATCH --error=results-%J.err
#SBATCH --output=results-%J.out
#SBATCH --job-name=MAN7916Setup
#SBATCH --gres=gpu:1

# This script is intended to be run by man7916launcher, not directly.

# load modules
module load cuda/cuda-12.6.0
module load python/python-3.11.4-oneapi-2023.1.0

# Get git repo with sparse checkout
cd ~/MAN7916
git init
git remote add origin https://github.com/amckenny/MAN7916.git
git sparse-checkout init --cone
git sparse-checkout set scripts corpora dictionaries
git pull origin main

# Unzip the Aussie corpus
cd corpora
mkdir -p aussie
unzip AussieTop100private.zip -d aussie
cd ..

# Set up virtual environment and install packages
source ~/MAN7916/.venv/bin/activate
python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install --upgrade -r requirements_newton.txt
python3 -m spacy download en_core_web_sm
python3 ./nltk_environ_setup.py
