#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o /hpc2hdd/home/dsaa2012_017/dl_test/logs/eval_%j.out
#SBATCH -e /hpc2hdd/home/dsaa2012_017/dl_test/logs/eval_%j.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py_312

echo "Starting evaluation pipeline at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

cd /hpc2hdd/home/dsaa2012_017/dl_test

mkdir -p logs eval_results/{masks,aligned,crops,metrics,cache}

# Phase 1: Pixel-level metrics (segmentation + alignment + SSIM/LPIPS/PSNR + CLIP)
# Process all rows in one go (single GPU)
echo "=== Phase 1: Preservation + Garment Metrics ==="
python eval_pipeline.py \
    --gpu 0 \
    --skip-vlm \
    --start 0 \
    --end -1

echo "Phase 1 done at $(date)"

# Phase 2: VLM scoring
echo "=== Phase 2: VLM Scoring ==="
python eval_pipeline.py \
    --gpu 0 \
    --vlm-only \
    --skip-metrics \
    --start 0 \
    --end -1

echo "Phase 2 done at $(date)"

# Phase 3: Aggregate results
echo "=== Phase 3: Aggregation ==="
python aggregate_results.py

echo "All done at $(date)"
