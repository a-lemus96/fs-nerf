#!/usr/bin/env bash # File: tests.sh

echo "Battery of Hyperparameter Tests"

params=(1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 5e-1) # Array of params

for lamb in ${params[*]}
do
	echo "($lamb)"
	python ../src/run-dsnerf.py --mu=1e-3 --use_viewdirs --n_iters=20000 --use_entropy --device_num=1 --lamb=$lamb
done
