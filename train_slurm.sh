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
#SBATCH --output=slurm_STTN2.out
##------------------------ End job description ------------------------

date
hostname
eval "$(conda shell.bash hook)"
conda activate AIRML
python train.py --config configs/twin_jet_develop_scaled_onlyrec.json --model sttn
##python train.py --config configs/davis.json --model sttn
##python test_custom.py --config configs/twin_jet.json --model sttn --ckpt 3000 --ds_name mics
date