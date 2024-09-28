#!/bin/bash
pwd=$(pwd)
cd best_models
for file in $pwd/best_models/*
do
   sbatch $file
done
