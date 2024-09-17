#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")


CUDA_VISIBLE_DEVICES=0,1,2,3 python main_instance_segmentation_stage2.py \
  general.experiment_name="train_stage2_${CURRENT_TIME}" \
  general.project_name="scannet" \
  optimizer.lr=0.0002 \
  data.batch_size=2 \
  data.num_workers=2 \
  trainer.max_epochs=50 \
  trainer.log_every_n_steps=5 \
  trainer.check_val_every_n_epoch=5 \
  general.train_mode=true \
  general.eval_on_segments=false \
  general.train_on_segments=false \
  model.num_queries=150 \
  matcher.cost_class=0.0 \
  general.topk_per_image=-1 \
  general.use_dbscan=false \
  general.gpus=4 \
  general.save_visualizations=False \
  general.checkpoint="PATH_TO_STAGE1_CHECKPOINT" \



