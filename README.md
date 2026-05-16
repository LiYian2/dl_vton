# Virtual Try-On Pipeline 

This repository contains a ComfyUI-based virtual try-on pipeline for garment-conditioned image editing. Given a target garment image and a person/character reference or prompt, the workflow generates a dressed character image while attempting to preserve person identity, body structure, and non-clothing regions. The project also includes an exploratory multi-view generation workflow for producing alternative viewing angles from a generated single-view result. The project was developed for local/HPC experimentation and for deployment on RunningHUB. The local workflows expose more debugging and multi-view functionality, while the RunningHUB workflow is simplified for online execution. 

## Group Members 
- Weifeng Chen
- Yuk Yeung Wong
- Boyi Zhang

## Repository Layout 
```text . 
├── Install.md # Detailed installation and HPC2 running instructions
├── environment.yml # Conda environment export used during development
├── workflows/ # RunningHUB-compatible workflow files
│   └── whole_workflow_runninghub.json
├── workflows_local/ # Local ComfyUI workflows for testing/debugging
└── README.md
``` 

## Pipeline Overview 
The full pipeline is organized around two usage modes. ### Single-view virtual try-on The single-view workflow takes a garment image and a person/character condition as input. It uses ComfyUI nodes for visual-language prompt extraction, segmentation/masking, inpainting or image editing, and image refinement. The intended output is a single edited try-on image in which the target garment is transferred onto the generated or referenced character. 
### Multi-view generation 
The local multi-view workflow is an exploratory extension. It uses the single-view try-on result and the target garment as visual references, then generates alternative views using view-conditioned prompts. This part is included as a bonus-style experiment and is not supported in the RunningHUB deployment because the required local nodes are unavailable online. 

## Installation
For full installation details, see [Install.md](Install.md). The project was tested mainly on HKUST(GZ) HPC2 with CUDA 12.8. The recommended local setup is: 

1. Create and activate the Python 3.12 Conda environment.
2. Install ComfyUI and ComfyUI-Manager.
3. Install the required ComfyUI custom nodes.
4. Download the required model checkpoints into the expected `ComfyUI/models/` subdirectories.
5. Launch ComfyUI and load the corresponding workflow JSON file.

A Conda environment export is provided in [environment.yml](environment.yml), but it may still require manual adjustment depending on the platform, CUDA version, and installed ComfyUI nodes. 

## Required Models 
The workflows expect several model families to be available under the ComfyUI model directory. Please refer to [Install.md](Install.md) for exact paths and file names. The main required components include: 
- Flux2-Klein diffusion checkpoints
- Qwen3-VL model files for visual-language prompt extraction
- Flux2-related LoRA checkpoints
- SAM3.1 checkpoint for segmentation/masking
- Flux2 text encoder
- Flux2 VAE

Because these model files are large and may have separate licenses or download restrictions, they are not included in this repository. 

## Usage 
### Option 1: Run locally with ComfyUI 
Use the workflows in [`workflows_local/`](workflows_local/) for local testing and debugging. These workflows may depend on custom nodes that are not available on RunningHUB. General steps: 

1. Start ComfyUI with the environment described in [Install.md](Install.md).
2. Open the ComfyUI web interface.
3. Load the desired JSON workflow from `workflows_local/`.
4. Set the input garment/person paths and output directory.
5. Run the workflow.

Some batch scripts or workflows assume fixed input/output directories. Please check the comments inside the workflow or script before running. For the multi-view workflow, the output from the first-stage single-view/character-neutralization workflow must be placed in the expected input directory for the second-stage generation workflow. 

### Option 2: Run on RunningHUB 
The workflow in [`workflows/whole_workflow_runninghub.json`](workflows/whole_workflow_runninghub.json) is designed for RunningHUB. You can either upload the workflow manually to RunningHUB or use the released application: [RunningHUB AI Application](https://www.runninghub.cn/post/2055684629914497025) 
Recommended usage: 
1. Choose **Run AI APP**.
2. Upload the required input images according to the interface instructions.
3. Run the workflow and inspect the final generated try-on result. Please ignore intermediate images shown during execution, because RunningHUB may expose intermediate previews after release. The workflow hides or disables some preview/save nodes to reduce unintended display of inappropriate intermediate results. RunningHUB usage must also follow its policy, including the prohibition of NSFW generation.

## Notes and Limitations 
- The RunningHUB version supports only the single-view workflow. 
- Multi-view generation is available only in the local workflow because it depends on local custom nodes.
- Local reproducibility depends heavily on the exact ComfyUI version, custom-node versions, CUDA version, and checkpoint placement.
- Some generated images may exhibit color drift, garment-detail inconsistency, boundary artifacts, or imperfect preservation of body structure, especially in challenging poses or multi-view settings.
- The workflows are designed for research/course-project experimentation rather than production deployment.

## Acknowledgements 
We thank the developers of ComfyUI and the broader open-source image-generation community for releasing the tools, custom nodes, workflows, and models that made this project possible. We also thank our instructors and classmates for their feedback and support throughout the project. The project uses or builds upon multiple ComfyUI custom nodes, including but not limited to: 
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 
- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) 
- [ComfyScript](https://github.com/Chaoses-Ib/ComfyScript) 
- [cg-use-everywhere](https://github.com/chrisgoringe/cg-use-everywhere) 
- [comfy-image-saver](https://github.com/giriss/comfy-image-saver) 
- [ComfyUI-Image-Saver](https://github.com/farizrifqi/ComfyUI-Image-Saver) 
- [ComfyUI_Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) 
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) 
- [ComfyUI_Custom_Nodes_AlekPet](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet) 
- [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials) 
- [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) 
- [ComfyUI_Qwen3-VL-Instruct](https://github.com/IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct) 
- [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL) 
- [ComfyUI-QwenVL-MultiImage](https://github.com/hardik-uppal/ComfyUI-QwenVL-MultiImage) 
- [Comfyui-QwenEditUtils](https://github.com/lrzjason/Comfyui-QwenEditUtils) 
- [ComfyUI-Easy-Sam3](https://github.com/yolain/ComfyUI-Easy-Sam3) 
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use) 
- [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) 
- [comfyui-inpaint-nodes](https://github.com/Acly/comfyui-inpaint-nodes) 
- [comfyui-impact-pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) 
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) 
- [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) 
- [ComfyUI-UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) 
- [ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger) 
- [ComfyUI-Logic](https://github.com/playboy-dongan/ComfyUI-Logic.git) 
- [ComfyUI-qwenmultiangle](https://github.com/jtydhr88/ComfyUI-qwenmultiangle.git) 
- [ComfyUI-CatVTON](https://github.com/pzc163/Comfyui-CatVTON) 
- [ComfyUI-IDM-VTON](https://github.com/TemryL/ComfyUI-IDM-VTON) 
- [IMAGDressing-ComfyUI](https://github.com/AIFSH/IMAGDressing-ComfyUI)

Some listed nodes or baseline integrations may be disabled in the final workflow but were used during experimentation. 
