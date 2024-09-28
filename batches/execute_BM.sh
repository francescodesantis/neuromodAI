#!/bin/bash
pwd=$(pwd)
cd best_models
for file in $pwd/*
do
   sbatch $file
done
