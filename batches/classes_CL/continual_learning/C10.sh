#!/bin/bash
#SBATCH --partition=boost_usr_prod
##SBATCH --array=1-100  # This submits 100 jobs, each with a different index
#SBATCH --account=IscrC_CATASTRO     # account name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00               # time limits: 1/2 hour
#SBATCH --error=C10/job.err            # standard error file
#SBATCH --output=C10/job.out           # standard output file
module load profile/deeplrn
module av cineca-ai
module load anaconda3

cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python continual_learning.py  --preset 6SoftHebbCnnCIFAR --resume all --model-name 'C10_2C_CL' --dataset-unsup CIFAR10_1 --dataset-sup CIFAR10_50 --continual_learning True --evaluate True --training-mode $1 --cf-sol $2 --head-sol $3 --top-k $4 --high-lr $5 --low-lr $6 --t-criteria $7 --delta-w-interval $8 --heads-basis-t $9 --selected-classes "${10}" --n-tasks "${11}" --evaluated-task "${12}" --classes-per-task "${13}" --topk-lock "${14}" --folder-id "${15}" --parent-f-id "${16}"

