#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=C10_Test/job.err            # standard error file
#SBATCH --output=C10_Test/job.out           # standard output file
#SBATCH --account=try24_antoniet     # account name
module load profile/deeplrn
#module av cineca-ai/modulepath
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python ray_search.py --preset 2SoftHebbCnnCIFAR --dataset-unsup CIFAR10_1 --dataset-sup CIFAR10_1 --folder-name 'CIFAR10_Test' --model-name 'CIFAR10_Test' --save-model --test True
