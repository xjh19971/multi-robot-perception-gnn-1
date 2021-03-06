#!/bin/bash
# Allows named arguments
set -k
python -u training.py -seed 1 -camera_idx 01234 -dataset warehouse-dgl -model_dir warehouse_models -model_file "model=multi_view_gcn2-bsize=8-lrt=0.005-camera_idx=01234-backbone=resnet50-seed=1-apply_noise_idx=None" -model multi_view_dgl -pretrained -backbone resnet50 -multi_gcn
for camera_idx in 01234; do
	for lrt in 0.005; do
		for batch_size in 8; do
			for seed in 1; do
				for apply_noise_idx in 0 01 012; do
					dataset="warehouse-noise-dgl"
					model_dir="warehouse_models"
					model="model=multi_view_gcn2-bsize="$batch_size"-lrt="$lrt"-camera_idx="$camera_idx"-backbone=resnet50-seed=1-apply_noise_idx="$apply_noise_idx""
					python -u training.py \
						-seed $seed \
						-camera_idx $camera_idx \
						-dataset $dataset \
						-model_dir $model_dir \
						-model_file $model \
						-model "multi_view_dgl"\
						-pretrained \
						-apply_noise_idx $apply_noise_idx\
						-backbone "resnet50"\
						-multi_gcn
				done
			done
		done
	done
done

