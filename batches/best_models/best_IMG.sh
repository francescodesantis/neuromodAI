#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=try24_antoniet     # account name
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1                    # 1 node
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00               # time limits: 1/2 hour
#SBATCH --error=IMG/job.err            # standard error file
#SBATCH --output=IMG/job.out           # standard output file
module load profile/deeplrn
module av cineca-ai
module load anaconda3

cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python ray_search.py --preset 6SoftHebbCnnImNet --dataset-unsup ImageNette_1 --dataset-sup ImageNette_200aug --folder-name 'ImageNette_SoftHebb6' --model-name 'Imagenette_SoftHebb6_Best' --save-model

