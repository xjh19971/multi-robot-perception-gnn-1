#!/bin/bash
# Allows named arguments
set -k
for model_num in 5 4 3 2 1; do
for camera_num in 5 4 3 2 1; do
        for batch_size in 1; do
                                python -u generate_feature.py \
                                        -model_file 'model=single_view-bsize=4-lrt=0.01-camera_num='$model_num'-seed=1' \
                                        -batch_size $batch_size \
                                        -camera_num $camera_num
                                sudo mv ./generated_data/* /media/data/dataset/airsim-map/airsim-mrmps-process/
                        done
                done
done
