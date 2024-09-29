#!/bin/bash
pwd=$(pwd)
rm -rf -d /g100_work/EIRI_E_POLIMI/rcasciot/neuromodAI/SoftHebb-main/Images/MULTD_CL
rm /g100_work/EIRI_E_POLIMI/rcasciot/neuromodAI/SoftHebb-main/MULTD_CL.json
cd full_datasets_CL
for file in $pwd/full_datasets_CL/*
do
   sbatch $file
done
