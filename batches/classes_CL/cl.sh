#!/bin/bash
pwd=$(pwd)

for file in $pwd/continual_learning/*
do
   sbatch $file
done
