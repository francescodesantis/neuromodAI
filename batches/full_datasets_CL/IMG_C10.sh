#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=best_models/IMG_C10/job.err            # standard error file
#SBATCH --output=best_models/IMG_C10/job.out           # standard output file
#SBATCH --account=EIRI_E_POLIMI     # account name
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
cp -r Training/results/hebb/result/network/ImageNette_Best Training/results/hebb/result/network/ImageNette_CIFAR10_CL
mkdir Training/results/hebb/result/network/ImageNette_CIFAR10_CL/models
mv Training/results/hebb/result/network/ImageNette_CIFAR10_CL/checkpoint.pth.tar Training/results/hebb/result/network/ImageNette_CIFAR10_CL/models/checkpoint.pth.tar
conda run -n softhebb python continual_learning.py --preset 6SoftHebbCnnImNet --resume all --model-name 'ImageNette_CIFAR10_CL' --dataset-unsup-1 ImageNette_1 --dataset-sup-1 ImageNette_200aug --dataset-unsup-2 CIFAR10_1 --dataset-sup-2 CIFAR10_50 --continual_learning True --skip-1 True --evaluate True

