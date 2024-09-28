#!/bin/bash
pwd=$(pwd)
for file in $pwd/*
do
   sbatch $pwd/$file
done
