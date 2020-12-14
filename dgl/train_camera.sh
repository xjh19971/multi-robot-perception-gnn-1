#!/bin/bash
# Allows named arguments
set -k
for camera_idx in 01234; do
    for lrt in 0.005; do
	for batch_size in 8; do
		for seed in 1; do
		    dataset="airsim"
                    model_dir="airsim_models"
		    model="model=single_view-bsize="$batch_size"-lrt="$lrt"-camera_idx="$camera_idx"-seed=1"
	       	    python -u training.py \
		    -seed $seed \
		    -camera_idx $camera_idx \
 		    -dataset $dataset \
		    -model_dir $model_dir \
		    -model_file $model \
		    -pretrained \
	    done
	    done
    done
    done
for camera_idx in 01234; do
	for lrt in 0.005; do
		for batch_size in 8; do
			for seed in 1; do
				dataset="airsim-noise"
				model_dir="airsim_noise_models"
				model="model=single_view-bsize="$batch_size"-lrt="$lrt"-camera_idx="$camera_idx"-seed=1"
				python -u training.py \
					-seed $seed \
					-camera_idx $camera_idx \
					-dataset $dataset \
					-model_dir $model_dir \
					-model_file $model \
					-pretrained \
				        -apply_noise_idx 1 	
			done
		done
	done
done
