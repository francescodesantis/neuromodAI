#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try24_antoniet     # account name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=CIFAR10/job.err            # standard error file
#SBATCH --output=CIFAR10/job.out           # standard output file
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python ray_search.py --preset 4SoftHebbCnnCIFAR --dataset-unsup CIFAR10_1 --dataset-sup CIFAR10_50 --folder-name 'CIFAR10_Best' --model-name 'CIFAR10_Best' --save-model
