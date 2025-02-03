#!/bin/bash
pwd=$(pwd)
rm -rf -d /leonardo_work/try24_antoniet/rcasciot/neuromodAI/SoftHebb-main/Images/TASKS_CL
for file in /leonardo_work/try24_antoniet/rcasciot/neuromodAI/batches/classes_CL/continual_learning/*
do
   sbatch $file
done
