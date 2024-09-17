#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
CURR_QUERY=400
CURR_TOPK=0.8
CURR_DBSCAN=0.05
CURR_DBSCAN_MIN_POINTS=5

CUDA_VISIBLE_DEVICES=0 python main_instance_segmentation.py \
    general.experiment_name="eval_${CURRENT_TIME}_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}_${CURR_DBSCAN_MIN_POINTS}" \
    general.project_name="scannetpp" \
    general.train_mode=false \
    general.eval_on_segments=true \
    general.train_on_segments=true \
    model.num_queries=${CURR_QUERY} \
    general.topk_per_image=${CURR_TOPK} \
    general.use_dbscan=true \
    general.dbscan_eps=${CURR_DBSCAN} \
    general.dbscan_min_points=${CURR_DBSCAN_MIN_POINTS} \
    general.gpus=1 \
    general.save_visualizations=false \
    general.checkpoint="checkpoints/segment3d.ckpt" \
    data.remove_small_group=15 \



