# Install Instructions
This file serves as the installation instruction on our Virtual Try On project on HKUST(GZ) HPC2. We do not guanrantee that it works on other platform.
## 1. Install Miniconda
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```
After installing Miniconda3 successfully, restart the bash, then
```bash
conda create -n py_312 python=3.12 -y
conda activate py_312
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
echo "python main.py --preview-method auto"
chmod +x run_gpu.sh
```

## 3. Run
```bash
srun -p debug -n 4 --mem=16G --gres=gpu:1 --time=00:30:00 --pty bash
cd path/conating/run_gpu.sh
chmod +x run_gpu.sh
./run_gpu.sh
```
