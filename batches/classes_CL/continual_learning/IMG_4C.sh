#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_CATASTRO     # account name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00               # time limits: 1/2 hour
#SBATCH --error=IMG_4C/job.err            # standard error file
#SBATCH --output=IMG_4C/job.out           # standard output file
module load profile/deeplrn
module av cineca-ai
module load anaconda3

cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python continual_learning.py --preset 6SoftHebbCnnImNet --resume all --model-name 'IMG_4C_CL' --dataset-unsup ImageNette_1 --dataset-sup ImageNette_50 --continual_learning True --evaluate True --classes 4 --training-mode $1 --cf-sol $2 --head-sol $3 --top-k $4 --high-lr $5 --low-lr $6 --t-criteria $7 --delta-w-interval $8 --heads-basis-t $9 --selected-classes "${10}"
