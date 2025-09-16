#!/bin/bash

# Set the features directory path
features_dir='/Users/nam.le/Desktop/research/SI-MIL/test_dataset/features'

# Run the training command
python train.py \
  --organ 'test_organ' \
  --dataset_split_path "$features_dir/train_dict.pickle" \
  --dataset_split_path_test "$features_dir/test_dict.pickle" \
  --dataset_split_deep_path "$features_dir/train_dict_deep.pickle" \
  --dataset_split_deep_path_test "$features_dir/test_dict_deep.pickle" \
  --features_deep_path "$features_dir/trainfeat_deep.pth" \
  --features_deep_path_test "$features_dir/testfeat_deep.pth" \
  --features_path "$features_dir/binned_hcf.csv" \
  --save_path "$features_dir/MIL_experiment" \
  --dropout_patch 0.4 \
  --num_epochs 40 \
  --weight_decay 0.01 \
  --lr 0.0002 \
  --top_k 20 \
  --use_additive 'yes' \
  --dropout_node 0.0 \
  --no_projection 'yes' \
  --feat_type 'patchattn_with_featattn_mlpmixer' \
  --stop_gradient 'no' \
  --cross_val_fold 0 \
  --temperature 3.0 \
  --percentile 0.75 \
  --torch_seed -1 