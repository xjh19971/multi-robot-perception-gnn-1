#!/bin/bash
# Allows named arguments
set -k
python -u eval_seg.py -seed 1 -camera_idx 01234 -dataset airsim-dgl -model_dir airsim_seg_models -model_file "best-model=multi_view_gcn_c-bsize=8-lrt=0.005-camera_idx=01234-backbone=mobilenetv2-seed=1-apply_noise_idx=None"
for camera_idx in 01234; do
	for lrt in 0.005; do
		for batch_size in 8; do
			for seed in 1; do
				for apply_noise_idx in 0; do
					dataset="airsim-noise-dgl"
					model_dir="airsim_seg_models"
					model="best-model=multi_view_gcn_c-bsize="$batch_size"-lrt="$lrt"-camera_idx="$camera_idx"-backbone=mobilenetv2-seed=1-apply_noise_idx="$apply_noise_idx""
					python -u eval_seg.py \
						-seed $seed \
						-camera_idx $camera_idx \
						-dataset $dataset \
						-model_dir $model_dir \
						-model_file $model \
						-apply_noise_idx $apply_noise_idx \
						-task seg
				done
			done
		done
	done
done

