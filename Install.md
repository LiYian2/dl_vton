# Installation Guide

This document describes the installation procedure used for the virtual try-on project on HKUST(GZ) HPC2. The reference environment uses Python 3.12 and CUDA 12.8. Other platforms may require changes to CUDA wheels, model paths, scheduler commands, and ComfyUI custom-node versions.

## 1. Create the Python Environment

Install Miniconda if it is not already available:

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

Restart the shell, then create and activate the project environment:

```bash
conda create -n py_312 python=3.12 -y
conda activate py_312
```

The repository also provides a development environment snapshot:

```bash
conda env create -f environment.yml
conda activate py_312
```

The exported environment is not guaranteed to be portable across CUDA versions or ComfyUI node revisions. If dependency conflicts occur, install the core ComfyUI requirements first, then install node-specific requirements incrementally.

## 2. Install ComfyUI

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

python -m pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git comfyui-manager
python -m pip install -r comfyui-manager/requirements.txt
```

On HPC2, the project uses the following ComfyUI root:

```text
/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI
```

Adjust scripts if your ComfyUI checkout or Miniconda installation is located elsewhere.

## 3. Install Custom Nodes

The easiest installation route on a workstation is to start ComfyUI with ComfyUI-Manager, load the workflow JSON, and install missing nodes through the manager interface. On HPC2, network policy can make manager-based installation unreliable; manual `git clone` installation under `ComfyUI/custom_nodes/` is therefore recommended.

The following custom-node inventory was verified from:

```text
/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI/custom_nodes
```

| Directory | Remote repository |
| --- | --- |
| `ComfyScript` | <https://github.com/Chaoses-Ib/ComfyScript.git> |
| `ComfyUI-Custom-Scripts` | <https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git> |
| `ComfyUI-Detail-Daemon` | <https://github.com/Jonseed/ComfyUI-Detail-Daemon.git> |
| `ComfyUI-Easy-Sam3` | <https://github.com/yolain/ComfyUI-Easy-Sam3.git> |
| `ComfyUI-Easy-Use` | <https://github.com/yolain/ComfyUI-Easy-Use.git> |
| `ComfyUI-IDM-VTON` | <https://github.com/TemryL/ComfyUI-IDM-VTON.git> |
| `ComfyUI-Inpaint-CropAndStitch` | <https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git> |
| `ComfyUI-KJNodes` | <https://github.com/kijai/ComfyUI-KJNodes.git> |
| `ComfyUI-Logic` | <https://github.com/playboy-dongan/ComfyUI-Logic.git> |
| `ComfyUI-QwenVL` | <https://github.com/1038lab/ComfyUI-QwenVL.git> |
| `ComfyUI-QwenVL-MultiImage` | <https://github.com/hardik-uppal/ComfyUI-QwenVL-MultiImage.git> |
| `ComfyUI-String-Helper` | <https://github.com/liuqianhonga/ComfyUI-String-Helper.git> |
| `ComfyUI-WD14-Tagger` | <https://github.com/pythongosssss/ComfyUI-WD14-Tagger.git> |
| `ComfyUI-WanVideoWrapper` | <https://github.com/kijai/ComfyUI-WanVideoWrapper.git> |
| `ComfyUI-llama-cpp_vlm` | <https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm> |
| `ComfyUI-qwenmultiangle` | <https://github.com/jtydhr88/ComfyUI-qwenmultiangle.git> |
| `ComfyUI-to-Python-Extension` | <https://github.com/pydn/ComfyUI-to-Python-Extension.git> |
| `ComfyUI-utils-nodes` | <https://github.com/zhangp365/ComfyUI-utils-nodes.git> |
| `ComfyUI_Comfyroll_CustomNodes` | <https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git> |
| `ComfyUI_Custom_Nodes_AlekPet` | <https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet.git> |
| `ComfyUI_LayerStyle` | <https://github.com/chflame163/ComfyUI_LayerStyle.git> |
| `ComfyUI_Qwen3-VL-Instruct` | <https://github.com/IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct.git> |
| `ComfyUI_UltimateSDUpscale` | <https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git> |
| `ComfyUI_essentials` | <https://github.com/cubiq/ComfyUI_essentials.git> |
| `Comfyui-QwenEditUtils` | <https://github.com/lrzjason/Comfyui-QwenEditUtils.git> |
| `cg-use-everywhere` | <https://github.com/chrisgoringe/cg-use-everywhere> |
| `comfy-image-saver` | <https://github.com/giriss/comfy-image-saver.git> |
| `comfyui-impact-pack` | <https://github.com/ltdrdata/ComfyUI-Impact-Pack> |
| `comfyui-inpaint-nodes` | <https://github.com/Acly/comfyui-inpaint-nodes> |
| `comfyui-manager` | <https://github.com/ltdrdata/ComfyUI-Manager> |
| `comfyui-various` | <https://github.com/jamesWalker55/comfyui-various> |
| `comfyui_controlnet_aux` | <https://github.com/Fannovel16/comfyui_controlnet_aux/> |
| `rgthree-comfy` | <https://github.com/rgthree/rgthree-comfy.git> |
| `was-node-suite-comfyui` | <https://github.com/WASasquatch/was-node-suite-comfyui/> |

Some installed repositories support baseline experiments, prompt utilities, or disabled exploratory workflows. The single-view workflow itself uses nodes from a smaller subset, including ComfyUI core nodes, ComfyUI_Qwen3-VL-Instruct or QwenVL-related nodes, ComfyUI_Comfyroll_CustomNodes, ComfyUI-Easy-Sam3, ComfyUI-Easy-Use, ComfyUI-KJNodes, ComfyUI_LayerStyle, comfyui-inpaint-nodes, rgthree-comfy, and image-saving utilities.

`llama-cpp-python` may need manual installation with CUDA-specific wheels:

```bash
conda activate py_312
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128
```

## 4. Download Required Models

All paths below are shown with `ComfyUI/` as the checkout root.

### 4.1 Diffusion Models

Create the Flux2-Klein model directory:

```bash
mkdir -p ComfyUI/models/diffusion_models/flux-2-klein
```

Required default checkpoint:

```text
ComfyUI/models/diffusion_models/flux-2-klein/Flux2-Klein-9B-True-v2-bf16.safetensors
```

Source:

<https://huggingface.co/wikeeyang/Flux2-Klein-9B-True-V2>

An alternative checkpoint used during experimentation is:

```text
ComfyUI/models/diffusion_models/flux-2-klein/F2K-9b-miracleinNSFWGeneration_10Fp8.safetensors
```

Source:

<https://civitai.com/models/2453960/miraclein-nsfw-generation-and-edit-flux2klein>

The default workflow uses `Flux2-Klein-9B-True-v2-bf16.safetensors`. Any use of alternative checkpoints should follow the checkpoint license, platform policy, and project safety constraints.

### 4.2 Visual-Language Model

Download Qwen3-VL-4B-Instruct:

<https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>

Place the model in the directories expected by the installed Qwen nodes:

```text
ComfyUI/models/LLM/Qwen-VL/Qwen3-VL-4B-Instruct
ComfyUI/models/prompt_generator/LLM/Qwen-VL/Qwen3-VL-4B-Instruct
ComfyUI/models/prompt_generator/Qwen3-VL-4B-Instruct
```

### 4.3 LoRA Checkpoints

Required LoRA:

```text
ComfyUI/models/loras/F2K_9bb-一致性consist_20260225.safetensors
```

Source:

<https://huggingface.co/weiqiang1978/Flux2Klein_Consistance_Edit_Lora>

Optional experimental LoRA:

```text
ComfyUI/models/loras/F2K_9b-破KLEIN-Unchained-V2.safetensors
```

### 4.4 SAM3.1

Create the SAM model directory and place the checkpoint there:

```text
ComfyUI/models/sam3/sam3.1_multiplex_fp16.safetensors
```

Source:

<https://huggingface.co/Comfy-Org/sam3.1>

### 4.5 Text Encoder

Required Flux2 text encoder:

```text
ComfyUI/models/text_encoders/qwen_3_8b_fp8mixed.safetensors
```

Source:

<https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b>

### 4.6 VAE

Required VAE:

```text
ComfyUI/models/vae/flux2-vae.safetensors
```

Source:

<https://huggingface.co/Comfy-Org/flux2-dev>

## 5. Launch ComfyUI on HPC2

Start an interactive GPU session:

```bash
ssh dsaa2012_017@hpc2login.hpc.hkust-gz.edu.cn
srun -p debug -n 8 --mem=32G --gres=gpu:1 --time=00:30:00 --pty bash
hostname
```

Record the allocated compute hostname, for example `gpu3-9`.

Launch ComfyUI on the compute node:

```bash
cd /hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI
source /hpc2hdd/home/dsaa2012_017/miniconda3/bin/activate py_312
module load cuda/12.8
python main.py --listen 0.0.0.0 --port 8188 --preview-method auto --enable-manager
```

From a local terminal, create an SSH tunnel through the login node:

```bash
ssh -L 18188:{compute_hostname}:8188 dsaa2012_017@hpc2login.hpc.hkust-gz.edu.cn
```

Open the ComfyUI interface at:

```text
http://localhost:18188
```

Load `workflows_local/whole_workflow.json` for local execution or `workflows/whole_workflow_runninghub.json` for the RunningHUB-compatible graph.

## 6. Batch Execution

The repository includes Slurm helper scripts for the HPC2 setup:

```bash
cd workflows_local
bash submit_vton_batch.sh
PERSON_IDX=0 CLOTH_START=0 CLOTH_END=15 bash submit_vton_single.sh
bash submit_multiview_batch.sh
```

Use dry-run mode to inspect generated Slurm scripts without submitting jobs:

```bash
DRY_RUN=1 bash submit_vton_batch.sh
DRY_RUN=1 PERSON_IDX=0 CLOTH_START=0 CLOTH_END=15 bash submit_vton_single.sh
DRY_RUN=1 bash submit_multiview_batch.sh
```

These scripts contain HPC2-specific paths and dataset assumptions. Before using them on another account or cluster, update `COMFYUI_ROOT`, input-folder names, output-folder names, and Conda activation paths in the corresponding Python and shell scripts.
