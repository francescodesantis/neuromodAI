#!/bin/bash
pwd=$(pwd)
rm /g100_work/EIRI_E_POLIMI/rcasciot/neuromodAI/SoftHebb-main/TASKS_CL.json
rm -rf -d /g100_work/EIRI_E_POLIMI/rcasciot/neuromodAI/SoftHebb-main/Images/TASKS_CL
cd continual_learning
for file in $pwd/continual_learning/*
do
   sbatch $file
done
