#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

TEST_SCENE=$1
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
CURR_QUERY=400
CURR_TOPK=-1
CURR_DBSCAN=0.05
CURR_DBSCAN_MIN_POINTS=5

CUDA_VISIBLE_DEVICES=0 python demo.py \
    general.experiment_name="eval_${TEST_SCENE}_${CURRENT_TIME}_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}_${CURR_DBSCAN_MIN_POINTS}" \
    general.project_name="demo" \
    general.train_mode=false \
    general.train_on_segments=true \
    model.num_queries=${CURR_QUERY} \
    general.topk_per_image=${CURR_TOPK} \
    general.use_dbscan=true \
    general.dbscan_eps=${CURR_DBSCAN} \
    general.dbscan_min_points=${CURR_DBSCAN_MIN_POINTS} \
    general.gpus=1 \
    general.save_visualizations=True \
    general.checkpoint="checkpoints/segment3d.ckpt" \
    general.test_scene=${TEST_SCENE} \
    data.remove_small_group=15 \
