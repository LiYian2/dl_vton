# It is just a reference of run_gpu.sh. You may need to change the directory of Miniconda3 and the path of ComfyUI.
#!/bin/bash
cd ComfyUI
source /hpc2hdd/home/dsaa2012_017/miniconda3/bin/activate py_312
module load cuda/12.8
#python main.py --preview-method auto
python main.py --listen 0.0.0.0 --port 8188 --preview-method auto --enable-manager
