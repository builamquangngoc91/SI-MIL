#!/bin/bash

#SBATCH --job-name=si-mil
#SBATCH --output=logs/inference.log
#SBATCH --nodelist=compute-0-5
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00

echo "Starting si-mil job on compute-0-5"

features_dir='path/to/test_dataset/features'

python inference.py 
    --dataset_split_path "$features_dir/train_dict.pickle" \
    --dataset_split_path_test "$features_dir/test_dict.pickle"  \
    --dataset_split_deep_path_test "$features_dir/test_dict_deep.pickle" \
    --features_deep_path_test "$features_dir/testfeat_deep.pth" \
    --features_path "$features_dir/binned_hcf.csv" \
    --model_weights_path "$features_dir/MIL_experiment/path/to/mil model weights" \
    --top_k 20 \
    --gpu_index 0 \ 
    --use_additive 'yes' \ 
    --temperature 3.0 \
    --percentile 0.75  \
    --no_projection 'yes' \
    --stop_gradient 'no' 

echo "Done"
