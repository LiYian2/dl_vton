# Virtual Try-On Pipeline

This repository contains a ComfyUI-based pipeline for garment-conditioned virtual try-on. Given a target garment image and a person or character reference, the workflow synthesizes a dressed image while preserving identity-related and non-garment visual attributes such as pose, face, hair, background, and global illumination. The repository also includes an experimental multi-view extension and an evaluation toolkit for comparing generated results against baseline methods.

The project was developed for course-project research and experimentation on HKUST(GZ) HPC2. The RunningHUB workflow is a simplified deployment variant, while the local workflows expose additional debugging, batching, and multi-view functionality.

## Contributors

- Weifeng Chen
- Yuk Yeung Wong
- Boyi Zhang

## Repository Structure

```text
.
├── README.md                         # Project overview and usage summary
├── Install.md                        # Installation, model placement, and HPC2 instructions
├── environment.yml                   # Conda environment export used during development
├── run_gpu.sh                        # Reference ComfyUI launch script for HPC2
├── workflows/
│   └── whole_workflow_runninghub.json # RunningHUB-compatible single-view workflow
├── workflows_local/
│   ├── whole_workflow.json           # Local single-view virtual try-on workflow
│   ├── multiview.json                # Experimental local multi-view workflow
│   ├── run_vton_batch.py             # ComfyScript batch runner for single-view VTON
│   ├── run_multiview_batch.py        # ComfyScript batch runner for multi-view generation
│   ├── submit_vton_batch.sh          # Slurm batch submission script
│   ├── submit_vton_single.sh         # Slurm single-range submission script
│   └── submit_multiview_batch.sh     # Slurm multi-view submission script
├── utils/
│   ├── color_refine.py               # Post-processing utility for color refinement
│   └── get_order.py                  # Utility script for ordering input cases
└── test/
    ├── README.md                     # Evaluation protocol documentation
    ├── eval_pipeline.py              # Quantitative and VLM-based evaluation pipeline
    ├── pairwise_vlm.py               # Pairwise VLM comparison script
    ├── aggregate_results.py          # Result aggregation script
    └── job_eval.sh                   # Slurm evaluation helper
```

The current structure separates deployment workflows, local research workflows, utility scripts, and evaluation code. This keeps the RunningHUB artifact independent from HPC-specific batch scripts and evaluation utilities.

## Method Overview

### Single-View Virtual Try-On

The primary workflow performs upper-garment transfer through a reference-conditioned image-editing pipeline. It combines visual-language garment description, SAM-based garment-region segmentation, Flux2-Klein generation, reference-latent conditioning, mask composition, and local refinement. The intended output is a single dressed image in which the target garment is transferred while non-garment regions remain as stable as possible.

### Multi-View Generation

The multi-view workflow is an exploratory extension. It uses the single-view result and garment reference to generate view-conditioned prompts and synthesize alternative camera views. This component is retained as a research extension and is not part of the RunningHUB deployment, because several required local nodes and batch assumptions are unavailable in the online environment.

### Evaluation

The `test/` directory contains an evaluation pipeline that combines masked preservation metrics, CLIP-based garment similarity, VLM absolute scoring, and pairwise VLM comparison. See [test/README.md](test/README.md) for the expected data layout and metric definitions.

## Installation

Detailed installation instructions are provided in [Install.md](Install.md). In brief, the recommended setup is:

1. Create a Python 3.12 Conda environment.
2. Install ComfyUI and ComfyUI-Manager.
3. Install the ComfyUI custom nodes listed in [Install.md](Install.md).
4. Place the required model checkpoints under the expected `ComfyUI/models/` subdirectories.
5. Launch ComfyUI and load the appropriate workflow JSON file.

The environment was tested primarily on HKUST(GZ) HPC2 with CUDA 12.8. The provided `environment.yml` is a development snapshot and may require adjustment on other systems.

## Required Model Families

The workflows require the following model families, all of which should be placed under the local ComfyUI model directory:

- Flux2-Klein diffusion checkpoint
- Qwen3-VL model files for visual-language prompt extraction
- Flux2-Klein LoRA checkpoint for consistency editing
- SAM3.1 checkpoint for segmentation
- Flux2 text encoder
- Flux2 VAE

Large checkpoints are not included in this repository. Their download sources, filenames, and target paths are listed in [Install.md](Install.md).

## Usage

### Local ComfyUI Execution

Use the workflows under `workflows_local/` for local or HPC2 experiments.

```bash
cd /hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI
source /hpc2hdd/home/dsaa2012_017/miniconda3/bin/activate py_312
module load cuda/12.8
python main.py --listen 0.0.0.0 --port 8188 --preview-method auto --enable-manager
```

Then open the ComfyUI interface, load `workflows_local/whole_workflow.json`, configure input paths, and run the workflow. For batch execution on HPC2, use the Slurm helpers in `workflows_local/` after adapting hard-coded dataset and ComfyUI paths if necessary.

### RunningHUB Execution

The workflow in `workflows/whole_workflow_runninghub.json` is designed for RunningHUB deployment. It can be uploaded manually or accessed through the released application:

[RunningHUB AI Application](https://www.runninghub.cn/post/2055684629914497025)

The RunningHUB version supports only the single-view workflow. Intermediate previews may be visible during execution depending on the platform interface; the final output should be used for assessment. Usage must follow RunningHUB platform policy and applicable model-license restrictions.

## Limitations

- The local and RunningHUB workflows are not identical, because RunningHUB does not provide every local custom node.
- Reproducibility depends on ComfyUI version, custom-node versions, CUDA version, checkpoint placement, and model quantization settings.
- The multi-view workflow is experimental and may suffer from identity drift, garment-detail inconsistency, and view-to-view geometry inconsistency.
- Generated results can contain color shifts, boundary artifacts, inaccurate garment details, or imperfect body-structure preservation, especially for unusual poses, transparent garments, or back-view inputs.
- The repository is intended for research and course-project experimentation rather than production deployment.

## Acknowledgements

This project builds on ComfyUI, ComfyUI-Manager, and multiple community-maintained ComfyUI custom nodes. The complete plugin inventory verified on the HPC2 environment is documented in [Install.md](Install.md). We thank the maintainers of these tools and the broader open-source image-generation community for making this work possible.
