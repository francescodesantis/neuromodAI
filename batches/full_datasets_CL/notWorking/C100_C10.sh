#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00               # time limits: 1/2 hour
#SBATCH --error=C100_C10/job.err            # standard error file
#SBATCH --output=C100_C10/job.out           # standard output file
#SBATCH --account=IscrC_CATASTRO     # account name
module load profile/deeplrn
module av cineca-ai
module load anaconda3

cd $WORK/rcasciot/neuromodAI/SoftHebb-main
rm -rf -d Training/results/hebb/result/network/CIFAR100_CIFAR10_CL
cp -r Training/results/hebb/result/network/CIFAR100_Best Training/results/hebb/result/network/CIFAR100_CIFAR10_CL
rm -rf -d Training/results/hebb/result/network/CIFAR100_CIFAR10_CL/models
mkdir Training/results/hebb/result/network/CIFAR100_CIFAR10_CL/models
cp Training/results/hebb/result/network/CIFAR100_CIFAR10_CL/checkpoint.pth.tar Training/results/hebb/result/network/CIFAR100_CIFAR10_CL/models/checkpoint.pth.tar
conda run -n softhebb python continual_learning.py --preset 4SoftHebbCnnCIFAR --resume all --model-name 'CIFAR100_CIFAR10_CL' --dataset-unsup-1 CIFAR100_1 --dataset-sup-1 CIFAR100_50 --dataset-unsup-2 CIFAR10_1 --dataset-sup-2 CIFAR10_50 --continual_learning True --skip-1 True --evaluate True 
