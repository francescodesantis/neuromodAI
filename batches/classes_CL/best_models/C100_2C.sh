#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_CATASTRO     # account name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00               # time limits: 1/2 hour
#SBATCH --error=C100_2C/job.err            # standard error file
#SBATCH --output=C100_2C/job.out           # standard output file
module load profile/deeplrn
module av cineca-ai
module load anaconda3

cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python ray_search_cl.py --preset 4SoftHebbCnnCIFAR --resume all --model-name 'C100_2C_Best' --dataset-unsup CIFAR100_1 --dataset-sup CIFAR100_50 --continual_learning True --evaluate True --classes 2

