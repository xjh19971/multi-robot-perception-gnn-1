#!/bin/bash
# Allows named arguments
set -k
gpu_idx=0
for camera_idx in 01234; do
	for lrt in 0.005; do
		for batch_size in 8; do
			for seed in 1; do
				for apply_noise_idx in 0; do
					dataset="airsim-noise"
					model_dir="airsim_depth_models"
					model="model=single_view-bsize="$batch_size"-lrt="$lrt"-camera_idx="$camera_idx"-backbone=mobilenetv2-seed=1-apply_noise_idx="$apply_noise_idx""
					python -u training.py \
						-seed $seed \
						-camera_idx $camera_idx \
						-dataset $dataset \
						-model_dir $model_dir \
						-model_file $model \
						-pretrained \
						-apply_noise_idx $apply_noise_idx\
						-backbone "mobilenetv2" \
						-task depth \
						-gpu_idx $gpu_idx
				done
			done
		done
	done
done
python -u training.py -seed 1 -camera_idx 01234 -dataset airsim -model_dir airsim_depth_models -model_file "model=single_view-bsize=8-lrt=0.005-camera_idx=01234-backbone=mobilenetv2-seed=1-apply_noise_idx=None" -pretrained -backbone mobilenetv2 -gpu_idx $gpu_idx -task depth

