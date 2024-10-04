#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try24_antoniet     # account name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=STL10_4C/job.err            # standard error file
#SBATCH --output=STL10_4C/job.out           # standard output file
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python ray_search_cl.py --preset 5SoftHebbCnnSTL --resume all --model-name 'STL10_4C_Best' --dataset-unsup STL10_1 --dataset-sup STL10_50 --continual_learning True --evaluate True --classes 4

