#!/bin/bash
#SBATCH --job-name="whitecans"
#SBATCH --partition="rocky"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-23:59
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --constraint=type_a

python -m scripts.pretrain_cans \
  --processed-dataset-dir ./data/c4_roberta_grouped_512 \
  --output-dir ./cans_pretrained \
  --num-train-epochs 1 \
  --iterations 5 \
  --momentum 0.9

# python -m scripts.pretrain_cans \
#   --dataset-name allenai/c4 \
#   --dataset-config en \
#   --dataset-split 'train[:1%]' \
#   --processed-dataset-dir ./data/c4_roberta_grouped_512 \
#   --preprocess-only \
#   --trust-remote-code