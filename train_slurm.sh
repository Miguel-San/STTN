#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --job-name=3_STTN_inpainting
#SBATCH --nodes=1
#SBATCH --nodelist=n008
#SBATCH --ntasks=1                 # number of MPI processes
#SBATCH --cpus-per-task=1           # number of CPUs per task
##SBATCH --tasks-per-node=1         # number of tasks per node
##SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --partition=gpu
#SBATCH --time=480:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miguel.sanchez.dominguez@upm.es
#SBATCH --output=slurm_STTN_sch.out
##------------------------ End job description ------------------------

date
hostname
eval "$(conda shell.bash hook)"
conda activate AIRML
##python train.py --config configs/twin_jet_old_scaled_dev.json --model sttn
##python test_custom.py --config configs/twin_jet_old_scaled_dev.json --model sttn --ckpt 4500 --ds_name full_ds_1e4

python train.py --config configs/scheduled_runs/w5_s5.json --model sttn
python train.py --config configs/scheduled_runs/w5_s10.json --model sttn
python train.py --config configs/scheduled_runs/w11_s1.json --model sttn
python train.py --config configs/scheduled_runs/w11_s5.json --model sttn
python train.py --config configs/scheduled_runs/w11_s10.json --model sttn
python train.py --config configs/scheduled_runs/w21_s1.json --model sttn
python train.py --config configs/scheduled_runs/w21_s5.json --model sttn
python train.py --config configs/scheduled_runs/w21_s10.json --model sttn

# python train.py --config configs/consistency_runs/twin_jet_old_scaled_c0.json --model sttn
# python train.py --config configs/consistency_runs/twin_jet_old_scaled_c1.json --model sttn
# python train.py --config configs/consistency_runs/twin_jet_old_scaled_c2.json --model sttn
# python train.py --config configs/consistency_runs/twin_jet_old_scaled_c3.json --model sttn
# python train.py --config configs/consistency_runs/twin_jet_old_scaled_c4.json --model sttn
# python train.py --config configs/consistency_runs/twin_jet_old_scaled_c5.json --model sttn
# python train.py --config configs/consistency_runs/twin_jet_old_scaled_c6.json --model sttn
# python train.py --config configs/consistency_runs/twin_jet_old_scaled_c7.json --model sttn
# python train.py --config configs/consistency_runs/twin_jet_old_scaled_c8.json --model sttn
# python train.py --config configs/consistency_runs/twin_jet_old_scaled_c9.json --model sttn

# python train.py --config configs/consistency_runs_10b/twin_jet_old_scaled_c0.json --model sttn
# python train.py --config configs/consistency_runs_10b/twin_jet_old_scaled_c1.json --model sttn
# python train.py --config configs/consistency_runs_10b/twin_jet_old_scaled_c2.json --model sttn
# python train.py --config configs/consistency_runs_10b/twin_jet_old_scaled_c3.json --model sttn
# python train.py --config configs/consistency_runs_10b/twin_jet_old_scaled_c4.json --model sttn
# python train.py --config configs/consistency_runs_10b/twin_jet_old_scaled_c5.json --model sttn
# python train.py --config configs/consistency_runs_10b/twin_jet_old_scaled_c6.json --model sttn
# python train.py --config configs/consistency_runs_10b/twin_jet_old_scaled_c7.json --model sttn
# python train.py --config configs/consistency_runs_10b/twin_jet_old_scaled_c8.json --model sttn
# python train.py --config configs/consistency_runs_10b/twin_jet_old_scaled_c9.json --model sttn

# python train.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c0.json --model sttn
# python train.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c1.json --model sttn
# python train.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c2.json --model sttn
# python train.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c3.json --model sttn
# python train.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c4.json --model sttn
# python train.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c5.json --model sttn
# python train.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c6.json --model sttn
# python train.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c7.json --model sttn
# python train.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c8.json --model sttn
# python train.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c9.json --model sttn

# python train.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c0.json --model sttn
# python train.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c1.json --model sttn
# python train.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c2.json --model sttn
# python train.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c3.json --model sttn
# python train.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c4.json --model sttn
# python train.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c5.json --model sttn
# python train.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c6.json --model sttn
# python train.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c7.json --model sttn
# python train.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c8.json --model sttn
# python train.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c9.json --model sttn

# python test_custom.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c0.json --model sttn --ckpt 20000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c1.json --model sttn --ckpt 20000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c2.json --model sttn --ckpt 20000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c3.json --model sttn --ckpt 20000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c4.json --model sttn --ckpt 20000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c5.json --model sttn --ckpt 20000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c6.json --model sttn --ckpt 20000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c7.json --model sttn --ckpt 20000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c8.json --model sttn --ckpt 20000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_3b_lr1E-5/twin_jet_old_scaled_c9.json --model sttn --ckpt 20000 --ds_name full_ds_1e4

# python test_custom.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c0.json --model sttn --ckpt 25000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c1.json --model sttn --ckpt 25000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c2.json --model sttn --ckpt 25000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c3.json --model sttn --ckpt 25000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c4.json --model sttn --ckpt 25000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c5.json --model sttn --ckpt 25000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c6.json --model sttn --ckpt 25000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c7.json --model sttn --ckpt 25000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c8.json --model sttn --ckpt 25000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs_1b_lr1E-5/twin_jet_old_scaled_c9.json --model sttn --ckpt 25000 --ds_name full_ds_1e4

# python test_custom.py --config configs/consistency_runs/twin_jet_old_scaled_c0.json --model sttn --ckpt 2000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs/twin_jet_old_scaled_c1.json --model sttn --ckpt 3500 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs/twin_jet_old_scaled_c2.json --model sttn --ckpt 4000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs/twin_jet_old_scaled_c3.json --model sttn --ckpt 5000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs/twin_jet_old_scaled_c4.json --model sttn --ckpt 4500 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs/twin_jet_old_scaled_c5.json --model sttn --ckpt 2000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs/twin_jet_old_scaled_c6.json --model sttn --ckpt 6000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs/twin_jet_old_scaled_c7.json --model sttn --ckpt 2000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs/twin_jet_old_scaled_c8.json --model sttn --ckpt 4000 --ds_name full_ds_1e4
# python test_custom.py --config configs/consistency_runs/twin_jet_old_scaled_c9.json --model sttn --ckpt 1500 --ds_name full_ds_1e4

# python test_custom.py --config configs/scheduled_runs/w11_s1.json --model sttn --ckpt 4500 --ds_name full_ds_1e4
# python test_custom.py --config configs/scheduled_runs/w21_s1.json --model sttn --ckpt 4000 --ds_name full_ds_1e4
# python test_custom.py --config configs/scheduled_runs/w11_s1_vg.json --model sttn --ckpt 1500 --ds_name full_ds_1e4

# python test_custom.py --config configs/scheduled_runs/twin_jet_old_scaled.json --model sttn --ckpt 4000 --ds_name full_ds_1e4
# python test_custom.py --config configs/scheduled_runs/twin_jet_old_scaled_vg.json --model sttn --ckpt 3500 --ds_name full_ds_1e4

# python test_custom.py --config configs/scheduled_runs/w5_s5.json --model sttn --ckpt 13000 --ds_name full_ds_1e4
# python test_custom.py --config configs/scheduled_runs/w5_s10.json --model sttn --ckpt 10000 --ds_name full_ds_1e4
# python test_custom.py --config configs/scheduled_runs/w11_s1.json --model sttn --ckpt 25000 --ds_name full_ds_1e4
# python test_custom.py --config configs/scheduled_runs/w11_s5.json --model sttn --ckpt 7600 --ds_name full_ds_1e4
# python test_custom.py --config configs/scheduled_runs/w11_s10.json --model sttn --ckpt 7600 --ds_name full_ds_1e4
# python test_custom.py --config configs/scheduled_runs/w21_s5.json --model sttn --ckpt 1000 --ds_name full_ds_1e4
# python test_custom.py --config configs/scheduled_runs/w21_s10.json --model sttn --ckpt 3500 --ds_name full_ds_1e4

date