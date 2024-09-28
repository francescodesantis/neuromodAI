#!/bin/bash
pwd=$(pwd)

for file in $pwd/best_models/*
do
   sbatch $file
done
