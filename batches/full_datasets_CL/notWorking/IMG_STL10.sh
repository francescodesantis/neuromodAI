#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try24_antoniet     # account name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=IMG_STL10/job.err            # standard error file
#SBATCH --output=IMG_STL10/job.out           # standard output file
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
rm -rf -d Training/results/hebb/result/network/ImageNette_STL10_CL
cp -r Training/results/hebb/result/network/ImageNette_Best Training/results/hebb/result/network/ImageNette_STL10_CL
rm -rf -d Training/results/hebb/result/network/ImageNette_STL10_CL/models
mkdir Training/results/hebb/result/network/ImageNette_STL10_CL/models
cp Training/results/hebb/result/network/ImageNette_STL10_CL/checkpoint.pth.tar Training/results/hebb/result/network/ImageNette_STL10_CL/models/checkpoint.pth.tar
conda run -n softhebb python continual_learning.py --preset 6SoftHebbCnnImNet --resume all --model-name 'ImageNette_STL10_CL' --dataset-unsup-1 ImageNette_1 --dataset-sup-1 ImageNette_200aug --dataset-unsup-2 STL10_1 --dataset-sup-2 STL10_50 --continual_learning True --skip-1 True --evaluate True

