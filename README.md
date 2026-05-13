# Project: Virtual Try-On Pipeline

## Group Members:
- Weifeng Chen
- Yuk Yeung Wong
- Boyi Zhang

## Installation Instructions for Local Testing:
1. Please check `Install.md`[Install.md] for detailed installation instructions.

## Usage Instructions:

1. The workflows in `workflows_local`[workflows_local] are designed for local testing and debugging. Some plugins are not available in the online version (RunningHUB). You can run them in local ComfyUI to see how they work. See `Installation.md` to set up local ComfyUI. To run the batch scripts, please make sure to set the correct paths for the input data and output directory in the scripts. The batch scripts will read the input data, run the workflows for each person, and save the generated images to the output directory. Please note that the batch scripts are designed for testing purposes and may not be optimized for performance. You can adjust the parameters in the workflows and batch scripts as needed for your specific use case. You also need to put the output of the first workflow (character neutralization) in the correct directory for the second workflow (multiview generation) to work properly. Please refer to the comments in the batch scripts for details on how to set up the input and output paths.

2. The workflows in `workflows_online`[workflows_online] are designed for RunningHUB. You can upload them to RunningHUB and run them there. Note that some plugins used in local workflows are not available in RunningHUB, so we have made some adjustments to the workflows for online use. Please refer to the comments in the workflow files for details. In addition, multiview generation is not supported in RunningHUB, so we have only included the workflow for single view generation in `workflows_online`. Please also note that it is prohibited to generate any NSFW content in RunningHUB, so we have hidden the preview node in the workflow to avoid generating NSFW content during character neutralization. 

## Acknowledgements:
- We would like to thank the developers of ComfyUI and the creators of the plugins we used in our workflows for providing such powerful tools for image generation and manipulation. We also want to thank our instructors and classmates for their support and feedback throughout the project.

### Plugins we used:

