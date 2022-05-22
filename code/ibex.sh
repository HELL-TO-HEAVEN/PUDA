#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --time=01:30:00
#SBATCH --mem=20G

python run.py
