#!/bin/bash
pwd=$(pwd)
cd continual_learning
for file in $pwd/continual_learning/*
do
   sbatch $file
done
