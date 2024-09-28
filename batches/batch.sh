#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1	    # 32 tasks per node
#SBATCH --cpus-per-task=1         
#SBATCH --time=00:10:00               # time limits: 1/2 hour
#SBATCH --error=job_test/job_test.err            # standard error file
#SBATCH --output=job_test/job_test.out           # standard output file
#SBATCH --account=EIRI_E_POLIMI     # account name
cd $WORK/rcasciot/neuromodAI/other
conda run -n softhebb python job_test.py
