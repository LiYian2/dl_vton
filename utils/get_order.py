import os
import glob
COMFYUI_ROOT = "/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI"
COMFYUI_INPUT = os.path.join(COMFYUI_ROOT, "input")
PERSON_FOLDER = "Designated Test_Character and clothes/Correspond"
def collect_images(folder_abs, extensions=None):
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    result = []
    for ext in extensions:
        for fp in glob.glob(os.path.join(folder_abs, f'*{ext}')) + \
                  glob.glob(os.path.join(folder_abs, f'*{ext.upper()}')):
            if os.path.isfile(fp):
                result.append(fp)
    return list(set(result))

person_folder_abs = os.path.join(COMFYUI_INPUT, PERSON_FOLDER)
person_images = collect_images(person_folder_abs)
person_images_rel = [os.path.relpath(p, COMFYUI_INPUT) for p in person_images]
person_images_sorted = sorted(person_images_rel)
print("Person images:", person_images_sorted)
