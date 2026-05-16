# Project: Virtual Try-On Pipeline

## Group Members:
- Weifeng Chen
- Yuk Yeung Wong
- Boyi Zhang

## Installation Instructions for Local Testing:
1. Please check [Install.md](Install.md) for detailed installation instructions.

## Usage Instructions:

1. The workflows in [workflows_local](workflows_local) are designed for local testing and debugging. Some plugins are not available in the online version (RunningHUB). You can run them in local ComfyUI to see how they work. See `Installation.md` to set up local ComfyUI. To run the batch scripts, please make sure to set the correct paths for the input data and output directory in the scripts. The batch scripts will read the input data, run the workflows for each person, and save the generated images to the output directory. Please note that the batch scripts are designed for testing purposes and may not be optimized for performance. You can adjust the parameters in the workflows and batch scripts as needed for your specific use case. You also need to put the output of the first workflow (character neutralization) in the correct directory for the second workflow (multiview generation) to work properly. Please refer to the comments in the batch scripts for details on how to set up the input and output paths.

2. The [workflow](workflows/whole_workflow_runninghub.json) is designed for RunningHUB.
+ You can upload them to RunningHUB and run them there. Note that some plugins used in local workflows are not available in RunningHUB, so we have made some adjustments to the workflows for online use.
+ Or you can visit [Our AI application on RunningHUB](https://www.runninghub.cn/post/2055684629914497025). It is suggested to choose `Run AI APP` and follow the instructions to upload the input images to try our workflow. Please ignore the intermediate iamges generated during the process as they can't be blocked after release.
+ Please also note that it is **prohibited** to generate any **NSFW** content in RunningHUB, so we have hidden the `PreviewImage` or `SaveImage` node in the workflow to avoid displaying or saving any NSFW content during character neutralization.

In addition, multiview generation is not supported in RunningHUB due to the lack of a specific node, so we have only included the workflow for single view generation in `workflows_online`. 

## Acknowledgements:
We would like to thank the developers of ComfyUI and the creators of the plugins we used in our workflows for providing such powerful tools for image generation and manipulation. We also want to express our sincere gratitude to the whole community who shared their workflows and models. Our instructors and classmates are also great, for their support and feedback throughout the project.

### Plugins we used:
- [cg-use-everywhere](https://github.com/chrisgoringe/cg-use-everywhere)
- [ComfyUI-Image-Saver](https://github.com/farizrifqi/ComfyUI-Image-Saver)
- [cg-use-everywhere](https://github.com/chrisgoringe/cg-use-everywhere)
- [comfy-image-saver](https://github.com/giriss/comfy-image-saver)
- [ComfyScript](https://github.com/Chaoses-Ib/ComfyScript)
- [ComfyUI_Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes)
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
- [ComfyUI_Custom_Nodes_AlekPet](https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet)
- [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials)
- [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)
- [ComfyUI_Qwen3-VL-Instruct](https://github.com/IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct)
- [ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale)
- [ComfyUI-CatVTON](https://github.com/pzc163/Comfyui-CatVTON)
- [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)
- [ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon)
- [ComfyUI-Easy-Sam3](https://github.com/yolain/ComfyUI-Easy-Sam3)
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)
- [ComfyUI-IDM-VTON](https://github.com/TemryL/ComfyUI-IDM-VTON)
- [comfyui-impact-pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
- [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch)
- [comfyui-inpaint-nodes](https://github.com/Acly/comfyui-inpaint-nodes)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm)
- [comfyui-manager](https://github.com/ltdrdata/ComfyUI-Manager)
- [Comfyui-QwenEditUtils](https://github.com/lrzjason/Comfyui-QwenEditUtils)
- [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL)
- [ComfyUI-QwenVL-MultiImage](https://github.com/hardik-uppal/ComfyUI-QwenVL-MultiImage)
- [ComfyUI-to-Python-Extension](https://github.com/pydn/ComfyUI-to-Python-Extension)
- [ComfyUI-utils-nodes](https://github.com/zhangp365/ComfyUI-utils-nodes)
- [comfyui-various](https://github.com/jamesWalker55/comfyui-various)
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- [ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)
- [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
- [was-node-suite-comfyui](https://github.com/WASasquatch/was-node-suite-comfyui)
- [ComfyUI-Logic ](https://github.com/playboy-dongan/ComfyUI-Logic.git)
- [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm)
- [ComfyUI-qwenmultiangle](https://github.com/jtydhr88/ComfyUI-qwenmultiangle.git)
- [comfyui-idm-vton (disabled)](https://github.com/TemryL/ComfyUI-IDM-VTON)
- [IMAGDressing-ComfyUI (disabled)](https://github.com/AIFSH/IMAGDressing-ComfyUI)
