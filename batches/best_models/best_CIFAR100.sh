#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=CIFAR100/job.err            # standard error file
#SBATCH --output=CIFAR100/job.out      # standard output file
#SBATCH --account=EIRI_E_POLIMI     # account name
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python ray_search.py --preset 4SoftHebbCnnCIFAR --dataset-unsup CIFAR100_1 --dataset-sup CIFAR100_50 --folder-name 'CIFAR100_SoftHebb4_Best' --model-name 'CIFAR100_SoftHebb4_Best' --save-model 
