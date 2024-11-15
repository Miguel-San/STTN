#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=3_STTN_inpainting
#SBATCH --nodes=1
#SBATCH --nodelist=n008
#SBATCH --ntasks=1                 # number of MPI processes
#SBATCH --cpus-per-task=10           # number of CPUs per task
##SBATCH --tasks-per-node=1         # number of tasks per node
##SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miguel.sanchez.dominguez@upm.es
#SBATCH --output=slurm_STTN.out
##------------------------ End job description ------------------------

date
hostname
eval "$(conda shell.bash hook)"
conda activate AIRML
python train.py --config configs/twin_jet_sens2_scaled.json --model sttn
##python test_custom.py --config configs/twin_jet_old_scaled.json --model sttn --ckpt 3500 --ds_name full_ds_1e4
date