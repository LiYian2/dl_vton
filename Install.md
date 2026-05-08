# Install Instructions
This file serves as the installation instruction on our Virtual Try On project on HKUST(GZ) HPC2. We do not guanrantee that it works on other platform. CUDA 12.8 is used for the project.
## 1. Install Miniconda
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```
After installing Miniconda3 successfully, restart the bash, then
```bash
conda env create -f environment.yml
conda activate py_312
# The below installation would cost a long time. Please makesure cuda is available.
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128
```
## 2. Install ComfyUI
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager
cd ..
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 #only cuda versions smaller or equal than 12.8 are installed in Module.
python -m pip install -r requirements.txt
python -m pip install -r custom_nodes/comfyui-manager/requirements.txt
cd ..
echo "#!/bin/bash" > run_gpu.sh
echo "cd ComfyUI" >> run_gpu.sh
echo "source {Path_Of_Miniconda3}/miniconda3/bin/activate py_312" >> run_gpu.sh # Please fill in the path
echo "module load cuda/12.8" >> run_gpu.sh
echo "python main.py --listen 0.0.0.0 --port 8188 --enable-manager"
chmod +x run_gpu.sh

echo "#!/bin/bash" > run_cpu.sh
echo "cd ComfyUI" >> run_cpu.sh
echo "source {Path_Of_Miniconda3}/miniconda3/bin/activate py_312" >> run_gpu.sh # Please fill in the path
echo "python main.py --listen 0.0.0.0 --port 8188 --cpu --enable-manager"
chmod +x run_cpu.sh
```

## 3. Install ComfyUI Plugins
The most easy way to install all the required plugins are through [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager). You may refer to `Run` part to open ComfyUI web page and open our workflow (*.json) and install all the required plugins accordingly. However, the network security policy on HPC-2 prohibits this. If so, please install manually accoding to their requirements.
They are:
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
- [comfyui-idm-vton (disabled)](https://github.com/TemryL/ComfyUI-IDM-VTON)
- [IMAGDressing-ComfyUI (disabled)](https://github.com/AIFSH/IMAGDressing-ComfyUI)

Notably, `llama-cpp-python` should be installed manually by compiling.
```bash
conda activate py_312
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128
```
If you encounter packages error, you may refer to our [environment.yml](environment.yml)

## 4. Download required models

### 4.1 Under `diffusion_models` folder
Download flux-2-klein checkpoints under ComfyUI/models/flux-2-klein/ . Say `ComfyUI/models/diffusion_models/flux-2-klein/F2K-9b-darkBeast_dbkBlitzV15_fp8.safetensors`
For flux-2-klein models, you need to download `Flux2-Klein-9B-True-v2-bf16.safetensors` and `F2K-9b-miracleinNSFWGeneration_10Fp8.safetensors`.

### 4.2 Under `LLM/Qwen-VL` folder
Download Qwen3-VL-4B-instrcut and put it under the folder, say `ComfyUI/models/LLM/Qwen-VL/Qwen3-VL-4B-Instruct`
Put this also under in `ComfyUI/models/prompt_generator/LLM/Qwen-VL/` and `ComfyUI/models/prompt_generator/`, say `ComfyUI/models/prompt_generator/LLM/Qwen-VL/Qwen3-VL-4B-Instruct` and `ComfyUI/models/prompt_generator/Qwen3-VL-4B-Instruct` respectively.

### 4.3 Under `loras` folder
Download `F2K_9bb-一致性consist_20260225.safetensors` and `F2K_9b-破KLEIN-Unchained-V2.safetensors`(optionally) and put it under, say `ComfyUI/models/loras/F2K_9bb-一致性consist_20260225.safetensors`

### 4.4 Under `sam3` folder 
If the folder doesn't exist, please create it first.
Doenload `sam3.1_multiplex_fp16.safetensors` and put it under the folder.

### 4.5 Under `text_encoders` folder 
Download `qwen_3_8b_fp8mixed.safetensors` and put it under the folder.

### 4.6 Under `vae` folder 
Doanload `flux2-vae.safetensors` and put it under the folder.

## 3. Run
For Terminal 1 (Run the task, start by `ssh dsaa2012_017@hpc2login.hpc.hkust-gz.edu.cn`)
```bash
srun -p debug -n 8 --mem=32G --gres=gpu:1 --time=00:30:00 --pty bash
hostname   # 记住这个，比如 gpu3-9
cd path/conating/run_gpu.sh
chmod +x run_gpu.sh
./run_gpu.sh
# or run it at the back
nohup ./run_gpu.sh 2>&1 &
# exit
pkill -f run_gpu.sh
```
For Terminal 2 (For port, start by `ssh -L 18188:{host_name}:8188 dsaa2012_017@hpc2login.hpc.hkust-gz.edu.cn`)
Then open `http://localhost:18188` in local browser.
