#!/bin/bash
pwd=$(pwd)
rm -rf -d /leonardo_work/try24_antoniet/rcasciot/neuromodAI/SoftHebb-main/Images/MULTD_CL
for file in /leonardo_work/try24_antoniet/rcasciot/neuromodAI/batches/full_datasets_CL/*
do
   sbatch $file
done
