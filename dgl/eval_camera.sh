#!/bin/bash
# Allows named arguments
set -k
for model_num in 5 4 3 2 1; do
	for batch_size in 1; do
			for seed in 1; do
				python -u eval.py \
					-seed $seed \
					-batch_size 1 \
					-model_file 'model=single_view-bsize=4-lrt=0.01-camera_num='$model_num'-seed=1'
			done
	done
done
