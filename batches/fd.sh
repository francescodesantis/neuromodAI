#!/bin/bash
pwd=$(pwd)
cd full_datasets_CL
for file in $pwd/full_datasets_CL/*
do
   sbatch $file
done
