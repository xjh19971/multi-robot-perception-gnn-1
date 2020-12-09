#!/bin/bash
# Allows named arguments
set -k
for camera_num in 5 4 3 2 1; do
	for batch_size in 4; do
		for lrt in 0.01; do
			for seed in 1; do
				python -u training.py \
					-seed $seed \
					-lrt $lrt \
					-batch_size $batch_size \
					-camera_num $camera_num \
					-pretrained \
					-multi_gpu
			done
		done
	done
done
