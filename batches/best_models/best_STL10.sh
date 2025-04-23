#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_CATASTRO     # account name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=STL10/job.err            # standard error file
#SBATCH --output=STL10/job.out           # standard output file
module load anaconda3
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python ray_search.py --preset 5SoftHebbCnnSTL --dataset-unsup STL10_unlabel --dataset-sup STL10_500aug --folder-name 'STL10_SoftHebb5_Best' --model-name 'STL10_SoftHebb5_Best' --save-model
