#!/bin/bash

#SBATCH --job-name=si-mil
#SBATCH --output=logs/inference.log
#SBATCH --nodelist=compute-0-1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1

echo "Starting si-mil job on compute-0-5"

python inference.py \
    --dataset_split_path "./LUNG/PathExpert/train_dict.pickle" \
    --dataset_split_path_test "./LUNG/PathExpert/test_dict.pickle"  \
    --dataset_split_deep_path_test "./LUNG/DINO/test_dict_deep.pickle" \
    --features_deep_path_test "./LUNG/DINO/testfeat_deep.pth" \
    --features_path "./LUNG/PathExpert/binned_hcf.csv" \
    --model_weights_path "MIL_experiment/result/checkpoint_best.pth" \
    --top_k 20 \
    --gpu_index 0 \
    --use_additive 'yes' \
    --temperature 3.0 \
    --percentile 0.75  \
    --no_projection 'yes' \
    --stop_gradient 'no' 

echo "Done"
