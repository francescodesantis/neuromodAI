#!/bin/bash
pwd=$(pwd)
cd full_datasets_CL
for file in $pwd/*
do
   sbatch $file
done
