#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00               # time limits: 1/2 hour
#SBATCH --error=best_models/Imagenette/job.err            # standard error file
#SBATCH --output=best_models/Imagenette/job.out           # standard output file
#SBATCH --account=EIRI_E_POLIMI     # account name
module load profile/deeplrn
module av cineca-ai
cd $WORK/rcasciot/neuromodAI/SoftHebb-main
conda run -n softhebb python ray_search.py --preset 6SoftHebbCnnImNet --dataset-unsup ImageNette_1 --dataset-sup ImageNette_200aug --folder-name 'ImageNette_SoftHebb6' --model-name 'Imagenette_SoftHebb6_Best' --save-model

