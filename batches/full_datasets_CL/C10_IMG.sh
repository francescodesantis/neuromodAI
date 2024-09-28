#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=best_models/C10_IMG/job.err            # standard error file
#SBATCH --output=best_models/C10_IMG/job.out           # standard output file
#SBATCH --account=EIRI_E_POLIMI     # account name
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
cp -r Training/results/hebb/result/network/CIFAR10_Best Training/results/hebb/result/network/CIFAR10_IMG_CL
mkdir Training/results/hebb/result/network/CIFAR10_IMG_CL/models
mv Training/results/hebb/result/network/CIFAR10_IMG_CL/checkpoint.pth.tar Training/results/hebb/result/network/CIFAR10_IMG_CL/models/checkpoint.pth.tar
conda run -n softhebb python continual_learning.py --preset 4SoftHebbCnnCIFAR --resume all --model-name 'CIFAR10_IMG_CL' --dataset-unsup-1 CIFAR10_1 --dataset-sup-1 CIFAR10_50 --dataset-unsup-2 ImageNette_1 --dataset-sup-2 ImageNette_50 --continual_learning True --skip-1 True --evaluate True 
