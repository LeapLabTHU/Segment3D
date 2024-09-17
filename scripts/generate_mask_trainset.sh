#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.05
CURR_DBSCAN_MIN_POINTS=5
CURR_TOPK=-1
CURR_QUERY=150

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")


CUDA_VISIBLE_DEVICES=0 python main_instance_segmentation_generate_mask_trainset.py \
    general.experiment_name="mask_generation_${CURRENT_TIME}" \
    general.train_mode=false \
    general.eval_on_segments=false \
    general.train_on_segments=false \
    model.num_queries=${CURR_QUERY} \
    general.topk_per_image=${CURR_TOPK} \
    general.use_dbscan=true \
    general.dbscan_eps=${CURR_DBSCAN} \
    general.dbscan_min_points=${CURR_DBSCAN_MIN_POINTS} \
    general.gpus=1 \
    general.save_visualizations=False \
    general.checkpoint="PATH_TO_STAGE1_CHECKPOINT" \
