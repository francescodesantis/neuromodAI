#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=C10_EV/job.err            # standard error file
#SBATCH --output=C10_EV/job.out           # standard output file
#SBATCH --account=EIRI_E_POLIMI     # account name
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
cp -r Training/results/hebb/result/network/CIFAR10_Best Training/results/hebb/result/network/C10_EV
rm -rf -d Training/results/hebb/result/network/C10_EV/models
mkdir Training/results/hebb/result/network/C10_EV/models
cp Training/results/hebb/result/network/C10_EV/checkpoint.pth.tar Training/results/hebb/result/network/C10_EV/models/checkpoint.pth.tar
conda run -n softhebb python multi_layer.py --preset 4SoftHebbCnnCIFAR --resume all --model-name 'C10_EV' --dataset-unsup CIFAR10_1 --dataset-sup CIFAR10_50 --evaluate True 
