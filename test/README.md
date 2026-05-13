# Virtual Try-On Evaluation Pipeline

Quantitative evaluation framework for Virtual Try-On methods. Supports:
- **Phase 1**: Pixel-level metrics (masked SSIM / LPIPS / PSNR) + CLIP garment similarity
- **Phase 2**: VLM absolute scoring (garment fidelity + human preference, 1-5)
- **Pairwise**: VLM pairwise win-rate comparison between methods

Uses SAM3.1 for person/garment segmentation, Qwen3-VL-4B for VLM scoring.

---

## Dependencies

```text
PyTorch 2.x + CUDA, transformers>=5.0, opencv-python, scikit-image, 
lpips, open_clip_torch, Pillow, numpy, tqdm, safetensors
```

External models (paths must be configured):
- SAM3.1 checkpoint (`sam3.1_multiplex_fp16.safetensors`)
  → loaded via `ComfyUI-Easy-Sam3` package (`sam3/model_builder.py`)
- Qwen3-VL-4B-Instruct (HuggingFace format)

---

## Input Data Structure

### 1. Person / Garment images

```
Designated Test_Character and clothes/
├── all char/           # Original person images (A)
│   ├── 1.jpg
│   ├── 00006_00.jpg
│   └── ...
└── Clothes/            # Target garment images (C)
    ├── 01.jpg
    ├── 010.jpg
    └── ...
```

### 2. Model outputs

Two naming conventions are supported:

**Baselines** (flat files): `{person_name}_{cloth_name}.{ext}`
```
baseline_outputs/
├── catvton/.../unpaired/00006_00_013.jpg
├── idmvton/5_09.jpg
└── ...
```

**Our model** (subfolder per sample): `{ours_person}_{cloth}/`
```
our_outputs/
└── 01_01/
    ├── 04_vton_raw_00001_.png    # Stage 1
    └── 07_final_00001_.png       # Stage 2
```

### 3. Person name mapping (`mapping.csv`)

If your model uses different person IDs than the ground-truth images:

```csv
ours_person_name, all_char_person_name
01,15
02,00006_00
...
```

`01` → ground-truth person `15.jpg`

### 4. Dataset CSV (`dataset.csv`)

Single file mapping every evaluation case:

```csv
sample_id,person_name,cloth_name,person_image_path,garment_image_path,model,model_output_path
sample_0000,32,014,all_char/32.jpg,Clothes/014.jpg,catvton,outputs/catvton/32_014.jpg
sample_0001,00006_00,013,all_char/00006_00.jpg,Clothes/013.jpg,idmvton,outputs/idmvton/00006_00_013.jpg
```

| Column | Description |
|--------|-------------|
| `sample_id` | Unique ID per (person, cloth) pair. Shared across methods for fair comparison |
| `person_name` | Ground-truth person filename stem (maps to `all_char/`) |
| `cloth_name` | Garment filename stem (maps to `Clothes/`) |
| `person_image_path` | Full path to original person image (A) |
| `garment_image_path` | Full path to target garment image (C) |
| `model` | Method name (e.g. `idmvton`, `our_model_v2_stage2`) |
| `model_output_path` | Full path to generated try-on result (B) |

---

## Running the Pipeline

### Phase 1: Pixel-level metrics (SAM3 segmentation + alignment + SSIM/LPIPS/PSNR + CLIP)

```bash
# Full run
python eval_pipeline.py --gpu 0 --skip-vlm

# Specific row range
python eval_pipeline.py --gpu 0 --skip-vlm --start 0 --end 6398

# Parallel chunks (for Slurm array jobs)
python eval_pipeline.py --gpu 0 --skip-vlm --chunk 0 --total-chunks 4 --start 0 --end 6398
```

Output files:
```
eval_results/metrics/
├── preservation_metrics_{tag}.csv    # Masked SSIM/LPIPS/PSNR per sample
└── garment_fidelity_metrics_{tag}.csv # CLIP garment similarity per sample
```

### Phase 2: VLM absolute scoring

```bash
python eval_pipeline.py --gpu 0 --vlm-only --skip-metrics --chunk 0 --total-chunks 4
```

Output files:
```
eval_results/metrics/
├── vlm_garment_scores_{tag}.csv      # 5 garment fidelity dimensions (1-5)
└── vlm_preference_scores_{tag}.csv   # 4 human preference dimensions (1-5)
```

### Pairwise VLM comparison

```bash
python pairwise_vlm.py --method-a idmvton --method-b our_model_v2_stage2 --chunk 0 --total-chunks 4
```

Output files:
```
eval_results/metrics/
└── pairwise_{method_a}_vs_{method_b}_chunkN.csv
```

Comparison protocol:
- Same person + cloth → 2 candidate outputs compared
- Candidates randomly assigned to "candidate_1" / "candidate_2" to avoid positional bias
- 4 dimensions: garment_fidelity, character_fidelity, harmony_realism, overall_preference
- Winner values: method name or "tie"

### Aggregation

```bash
python aggregate_results.py
```

Computes mean ± std per method across all chunks, produces summary tables.

---

## Output Metrics

### Phase 1 CSV columns

```text
sample_id, person_name, cloth_name, model,
alignment_status, A_original_size, B_original_size,
valid_area_ratio_person_noncloth, valid_area_ratio_full_noncloth,
Masked_SSIM_person_noncloth, Masked_LPIPS_person_noncloth, Masked_PSNR_person_noncloth,
Masked_SSIM_full_noncloth, Masked_LPIPS_full_noncloth, Masked_PSNR_full_noncloth,
CLIP_Garment_Similarity
```

### Phase 2 VLM CSV columns

```text
sample_id, model,
VLM_garment_fidelity, VLM_character_fidelity, VLM_harmony_realism, VLM_overall_quality
```

### Pairwise CSV columns

```text
sample_id, person_name, cloth_name,
swap, candidate_1_method, candidate_2_method,
garment_fidelity_winner, garment_fidelity_reason,
character_fidelity_winner, character_fidelity_reason,
harmony_realism_winner, harmony_realism_reason,
overall_preference_winner, overall_preference_reason,
vlm_raw, error
```

---

## Metric Definitions

### Masked SSIM / LPIPS / PSNR

Computed only on **person non-clothing region** after canonical alignment:
1. SAM3.1 segments person mask + upper-clothes mask for both A (original) and B (generated)
2. Both images independently aligned to 768×1024 canvas based on person bbox (scale + translation only, no warping)
3. Union clothing mask dilated → evaluation mask = valid pixels ∩ person region ∖ clothing
4. Pixel-level metrics averaged only over evaluation mask

### CLIP Garment Similarity

Cosine similarity between CLIP embeddings of:
- Target garment image (C)
- Cropped garment region from generated output (B)

### VLM Absolute Scoring (1-5)

Two separate queries to Qwen3-VL-4B:
- **Garment fidelity**: category, color, pattern, shape, overall
- **Human preference**: garment_fidelity, character_fidelity, harmony_realism, overall_quality

### Pairwise Win Rate

For each (person, cloth, method_i, method_j):
- VLM compares two candidates blindly (randomized order)
- Reports winner per dimension
- Win rate = #wins / #valid comparisons

---

## Slurm Submission (HPC)

### Phase 1 (pixel metrics, GPU + SAM3)

```bash
#!/bin/bash
#SBATCH -p emergency_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3

python eval_pipeline.py --gpu 0 --skip-vlm --chunk ${SLURM_ARRAY_TASK_ID} --total-chunks 4
```

### Phase 2 / Pairwise (VLM only, GPU)

```bash
#!/bin/bash
#SBATCH -p emergency_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3

python pairwise_vlm.py --method-a idmvton --method-b ours_stage2 --chunk ${SLURM_ARRAY_TASK_ID} --total-chunks 4
```

---

## Configuration

Key constants in `eval_pipeline.py`:

```python
CANVAS_WIDTH = 768
CANVAS_HEIGHT = 1024
TARGET_PERSON_HEIGHT_RATIO = 0.86    # Person occupies 86% of canvas height
TARGET_CENTER_X_RATIO = 0.50
TARGET_CENTER_Y_RATIO = 0.52
PAD_VALUE = 127                        # Gray padding
SAM_CONFIDENCE_THRESHOLD = 0.3         # Mask confidence threshold
DILATION_PX_BASE = 8                   # Cloth union dilation
DILATION_RATIO = 0.015                 # Cloth union dilation relative to person height
```

Paths in `eval_pipeline.py` and `pairwise_vlm.py`:

```python
SAM3_CKPT = "/path/to/sam3.1_multiplex_fp16.safetensors"
VLM_PATH = "/path/to/Qwen3-VL-4B-Instruct"
DATASET_CSV = "/path/to/dataset.csv"
```

---

## Notes

- **Canonical alignment** uses only global scale + translation (person bbox → canvas center). No non-rigid warping or pose-based alignment — this preserves real model failures.
- **File naming**: chunked results use `_{chunk_tag}.csv` suffix. The tag format is `chunk{chunk}_{start}_{end}` to prevent collisions across different runs.
- **VLM reproducibility**: temperature=0, do_sample=False for deterministic outputs.
- **Person mapping**: If your model's person IDs differ from baseline person IDs, provide `mapping.csv` and update `dataset.csv` accordingly.
- **Multi-GPU**: The pipeline doesn't natively support multi-GPU, but chunked array jobs provide natural parallelism across nodes.
