#!/bin/bash
#SBATCH --job-name=prism-rl
#SBATCH --partition=cpunodes
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH --time=5:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=asher.arya@mail.utoronto.ca

eval "$(conda shell.bash hook)"
conda activate prism3

cd ~/coding/PRISM-LLM

python PRISM-Guided-Learning/src/Evaluator.py
