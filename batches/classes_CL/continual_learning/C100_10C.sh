#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=C100_10C/job.err            # standard error file
#SBATCH --output=C100_10C/job.out           # standard output file
#SBATCH --account=EIRI_E_POLIMI     # account name
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python continual_learning.py --preset 4SoftHebbCnnCIFAR --resume all --model-name 'C100_10C_CL' --dataset-unsup CIFAR100_1 --dataset-sup CIFAR100_50 --continual_learning True --evaluate True --classes 10

