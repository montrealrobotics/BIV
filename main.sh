#!/bin/bash
#SBATCH -o /home/[organization]/k/[username]/logs/noise_%j.out   # Change this!
#SBATCH --cpus-per-task=4  
#SBATCH --gres=gpu:1        
#SBATCH --mem=32Gb    

# Load cuda
module load cuda/10.0
# 1. You have to load singularity
module load singularity
# 2. Then you copy the container to the local disk
rsync -avz /home/[organization]/k/[username]/environments/pytorch_f.simg $SLURM_TMPDIR     # Change this!
# 3. Copy your dataset on the compute node
rsync -avz /network/tmp1/[username]/ $SLURM_TMPDIR        # Change this!
# 3.1 export wandb api key
export WANDB_API_KEY= "put your wandb key here"       # Change this!
# 4. Executing your code with singularity
singularity exec --nv -H $HOME:/home/ -B $SLURM_TMPDIR:/datasets/ -B $SLURM_TMPDIR:/final_outps/  $SLURM_TMPDIR/pytorch_f.simg python ~/apps/IV_RL_server/main.py --exp_settings=$1 --noise_settings=$2 --noise_params=$3 --estim_noise_params=$4
# 5. Move results back to the login node.
rsync -avz $SLURM_TMPDIR --exclude="Datasets" --exclude="pytorch_f.simg"  /home/[organization]/k/[username]/outputs  # Change this!
