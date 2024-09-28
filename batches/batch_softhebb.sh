#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00               # time limits: 1/2 hour
#SBATCH --error=job_test/job_test.err            # standard error file
#SBATCH --output=job_test/job_test.out           # standard output file
#SBATCH --account=EIRI_E_POLIMI     # account name
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
python continual_learning.py --preset 2SoftHebbCnnCIFAR --resume all --model-name '2SoftHebbCnnCIFAR' --dataset-unsup-1 STL10_1 --dataset-sup-1 STL10_1 --dataset-unsup-2 ImageNette_1 --dataset-sup-2 ImageNette_1 --continual_learning True --evaluate True 
