#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")


CUDA_VISIBLE_DEVICES=0,1 python main_instance_segmentation_stage1.py \
  general.experiment_name="train_stage1_${CURRENT_TIME}" \
  general.project_name="scannet" \
  optimizer.lr=0.0002 \
  data.batch_size=8 \
  data.num_workers=4 \
  trainer.max_epochs=20 \
  trainer.log_every_n_steps=5 \
  trainer.val_check_interval=2000 \
  general.save_visualizations=False \
  general.gpus=2 \
  model.num_queries=100 \
