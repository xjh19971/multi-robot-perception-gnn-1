#!/bin/bash
# Allows named arguments
set -k
gpu_idx=0
python -u training.py -seed 1 -camera_idx 01234 -dataset airsim-dgl -model_dir airsim_depthseg_models -model_file "model=multi_view_gcn-bsize=8-lrt=0.005-camera_idx=01234-backbone=mobilenetv2-seed=1-apply_noise_idx=None" -model multi_view_dgl -pretrained -backbone mobilenetv2 -gpu_idx $gpu_idx -task depthseg
for camera_idx in 01234; do
	for lrt in 0.005; do
		for batch_size in 8; do
			for seed in 1; do
				for apply_noise_idx in 0; do
					dataset="airsim-noise-dgl"
					model_dir="airsim_depthseg_models"
					model="model=multi_view_gcn-bsize="$batch_size"-lrt="$lrt"-camera_idx="$camera_idx"-backbone=mobilenetv2-seed=1-apply_noise_idx="$apply_noise_idx""
					python -u training.py \
						-seed $seed \
						-camera_idx $camera_idx \
						-dataset $dataset \
						-model_dir $model_dir \
						-model_file $model \
						-model "multi_view_dgl"\
						-pretrained \
						-apply_noise_idx $apply_noise_idx\
						-backbone "mobilenetv2" \
						-task depthseg \
						-gpu_idx $gpu_idx
				done
			done
		done
	done
done

