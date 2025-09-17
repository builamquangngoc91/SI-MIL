#!/bin/bash

#SBATCH --job-name=si-mil
#SBATCH --output=logs/train.log
#SBATCH --nodelist=compute-0-5
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00

echo "Starting si-mil job on compute-0-5"


python train.py \
    --organ 'test_organ' \
    --dataset_split_path "./LUNG/PathExpert/train_dict.pickle" \
    --dataset_split_path_test "./LUNG/PathExpert/test_dict.pickle"  \
    --dataset_split_deep_path "./LUNG/DINO/train_dict_deep.pickle" \
    --dataset_split_deep_path_test "./LUNG/DINO/test_dict_deep.pickle" \
    --features_deep_path "./LUNG/DINO/trainfeat_deep.pth"  \
    --features_deep_path_test "./LUNG/DINO/testfeat_deep.pth" \
    --features_path "./LUNG/PathExpert/binned_hcf.csv" \
    --save_path "./MIL_experiment" \
    --dropout_patch 0.4 \
    --num_epochs 40 \
    --weight_decay 0.01 \
    --lr 0.0002 \
    --top_k 20 \
    --gpu_index 0  \
    --use_additive 'yes'  \
    --dropout_node 0.0  \
    --no_projection 'yes' \
    --feat_type 'patchattn_with_featattn_mlpmixer'  \
    --stop_gradient 'no' \
    --cross_val_fold 0 \
    --temperature 3.0 \
    --percentile 0.75 \
    --torch_seed -1

echo "Done"

