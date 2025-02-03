#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try24_antoniet     # account name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00               # time limits: 1/2 hour
#SBATCH --error=C10_IMG/job.err            # standard error file
#SBATCH --output=C10_IMG/job.out           # standard output file
module load profile/deeplrn
module av cineca-ai
module load anaconda3

cd $WORK/rcasciot/neuromodAI/SoftHebb-main
rm -rf -d Training/results/hebb/result/network/CIFAR10_IMG_CL
cp -r Training/results/hebb/result/network/CIFAR10_Best Training/results/hebb/result/network/CIFAR10_IMG_CL
rm -rf -d Training/results/hebb/result/network/CIFAR10_IMG_CL/models
mkdir Training/results/hebb/result/network/CIFAR10_IMG_CL/models
cp Training/results/hebb/result/network/CIFAR10_IMG_CL/checkpoint.pth.tar Training/results/hebb/result/network/CIFAR10_IMG_CL/models/checkpoint.pth.tar
conda run -n softhebb python continual_learning.py --preset 4SoftHebbCnnCIFAR --resume all --model-name 'CIFAR10_IMG_CL' --dataset-unsup-1 CIFAR10_1 --dataset-sup-1 CIFAR10_50 --dataset-unsup-2 ImageNette_1 --dataset-sup-2 ImageNette_50 --continual_learning True --skip-1 True --evaluate True --training-mode $1 --cf-sol $2 --head-sol $3 --top-k $4 --high-lr $5 --low-lr $6 --t-criteria $7 --delta-w-interval $8 --heads-basis-t $9 --selected-classes "${10}"
