#!/usr/bin/env python3
"""
Color Refinement Pipeline: MKL (Monge-Kantorovitch Linear) color transfer.
Step 1 (08_env_color): Non-garment color correction using 00_input_char as reference.
Step 2 (09_cloth_color): Garment-level color correction using pure cloth as reference.

References:
  - color-matcher MKL solver (https://github.com/hahnec/color-matcher/)
  - SAM3.1 via ComfyUI-Easy-Sam3 custom node
"""

import os, sys, argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import cv2

# ── SAM3 paths ──────────────────────────────────────────────────────
SAM3_PATH = "/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI/custom_nodes/ComfyUI-Easy-Sam3"
sys.path.insert(0, SAM3_PATH)
SAM3_CKPT = "/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI/models/sam3/sam3.1_multiplex_fp16.safetensors"

# ── Data paths ──────────────────────────────────────────────────────
OUTPUT_DIR = Path("/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI/output/final")
PURE_CLOTH_DIR = Path("/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI/input/Designated Test_Character and clothes/Pure Clothes")

# ── Parameters ──────────────────────────────────────────────────────
LAMBDA_NG = 0.3
LAMBDA_G = 0.4
DILATION_PX_BASE = 8
DILATION_RATIO = 0.015
BLUR_SIGMA_RATIO = 0.01
SAM_CONFIDENCE_THRESHOLD = 0.3


def pil_to_np(img):
    return np.array(img)


# ── MKL Color Transfer ───────────────────────────────────────────────
def mkl_transfer(src_img, ref_img, src_mask=None, ref_mask=None, strength=1.0):
    """
    MKL color transfer from ref_img to src_img, with optional per-pixel masks
    for statistics estimation. Implementation follows color-matcher's mkl_solver.

    Args:
        src_img:  (H,W,3) RGB uint8 — image to transform
        ref_img:  (H',W',3) RGB uint8 — color reference
        src_mask: (H,W) bool — True pixels used for src statistics
        ref_mask: (H',W') bool — True pixels used for ref statistics
        strength: [0,1] transfer strength

    Returns:
        (H,W,3) RGB uint8
    """
    src_f = src_img.astype(np.float64)
    ref_f = ref_img.astype(np.float64)

    H, W = src_f.shape[:2]

    # Flatten pixels: [3, N]
    src_flat = src_f.reshape(-1, 3).T
    ref_flat = ref_f.reshape(-1, 3).T

    # Apply masks for statistics
    if src_mask is not None:
        src_mask_flat = src_mask.reshape(-1)
        src_pix = src_flat[:, src_mask_flat]  # [3, N_mask_src]
    else:
        src_pix = src_flat

    if ref_mask is not None:
        ref_mask_flat = ref_mask.reshape(-1)
        ref_pix = ref_flat[:, ref_mask_flat]  # [3, N_mask_ref]
    else:
        ref_pix = ref_flat

    if src_pix.shape[1] < 10 or ref_pix.shape[1] < 10:
        return src_img

    # Compute covariance matrices (same as np.cov)
    cov_src = np.cov(src_pix)
    cov_ref = np.cov(ref_pix)

    # Mean vectors: [3, 1]
    mu_src = src_pix.mean(axis=1)[:, np.newaxis]
    mu_ref = ref_pix.mean(axis=1)[:, np.newaxis]

    # ── MKL solver (from color-matcher) ──
    eig_val_r, eig_vec_r = np.linalg.eig(cov_src)
    eig_val_r[eig_val_r < 0] = 0
    val_r = np.diag(np.sqrt(eig_val_r[::-1]))
    vec_r = np.array(eig_vec_r[:, ::-1])
    inv_r = np.diag(1.0 / (np.diag(val_r + np.spacing(1))))

    mat_c = val_r @ vec_r.T @ cov_ref @ vec_r @ val_r
    eig_val_c, eig_vec_c = np.linalg.eig(mat_c)
    eig_val_c[eig_val_c < 0] = 0
    val_c = np.diag(np.sqrt(eig_val_c))

    T = vec_r @ inv_r @ eig_vec_c @ val_c @ eig_vec_c.T @ inv_r @ vec_r.T

    # Apply transformation to all source pixels
    transformed = np.dot(T, src_flat - mu_src) + mu_ref  # [3, H*W]
    transformed = transformed.T.reshape(H, W, 3)          # [H, W, 3]

    # Strength interpolation (same as color-matcher ComfyUI node)
    if strength < 1.0:
        transformed = src_f + strength * (transformed - src_f)

    return np.clip(transformed, 0, 255).astype(np.uint8)


# ── SAM3 Segmentation ───────────────────────────────────────────────
class SAM3Segmentation:
    def __init__(self, device='cuda'):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print("Loading SAM3.1 model...")
        self.model = build_sam3_image_model(
            device=device,
            load_from_HF=False,
            checkpoint_path=SAM3_CKPT,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        self.model = self.model.float()
        if device.startswith('cuda'):
            self.model = self.model.cuda()

        param_devices = set()
        for p in self.model.parameters():
            param_devices.add(str(p.device))
        print(f"  Model param devices: {param_devices}")

        self.processor = Sam3Processor(
            self.model, resolution=1008, device=device,
            confidence_threshold=SAM_CONFIDENCE_THRESHOLD
        )
        self.device = device
        self._torch = torch
        print("  SAM3.1 loaded.")

    def segment(self, image, prompt):
        torch = self._torch
        with torch.no_grad():
            state = {}
            state = self.processor.set_image(image, state)
            state = self.processor.set_text_prompt(prompt, state)

            masks = state.get("masks")
            boxes = state.get("boxes")
            scores = state.get("scores")

            if masks is None:
                return {'masks': np.zeros((0, *image.size[::-1]), dtype=bool),
                        'boxes': np.zeros((0, 4), dtype=np.float32),
                        'scores': np.zeros((0,), dtype=np.float32)}

            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
                boxes = boxes.cpu().numpy() if boxes is not None else np.zeros((masks.shape[0], 4))
                scores = scores.cpu().numpy() if scores is not None else np.zeros(masks.shape[0])

            if masks.ndim == 4:
                masks = masks[:, 0]
            return {'masks': masks, 'boxes': boxes, 'scores': scores}

    def get_person_mask(self, image):
        result = self.segment(image, "person")
        masks = result['masks']
        scores = result['scores']

        if len(masks) == 0:
            return None

        img_h, img_w = image.size[::-1]
        img_area = img_h * img_w
        best_mask = None
        best_score = -1.0

        for i in range(len(masks)):
            mask = masks[i] > 0.5 if masks[i].dtype != bool else masks[i]
            area = mask.sum()
            area_ratio = area / img_area
            if area_ratio < 0.01 or area_ratio > 0.95:
                continue
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            cy, cx = ys.mean(), xs.mean()
            center_dist = np.sqrt((cy / img_h - 0.5) ** 2 + (cx / img_w - 0.5) ** 2)
            s = area * (1.0 - center_dist) * float(scores[i])
            if s > best_score:
                best_score = s
                best_mask = mask

        if best_mask is None and len(masks) > 0:
            idx = int(np.argmax(scores))
            best_mask = masks[idx] > 0.5 if masks[idx].dtype != bool else masks[idx]
        return best_mask

    def get_upper_clothes_mask(self, image, person_mask=None):
        result = self.segment(image, "upper clothes")
        masks = result['masks']

        if len(masks) == 0:
            return None

        img_h, img_w = image.size[::-1]
        img_area = img_h * img_w
        kept_masks = []

        for i in range(len(masks)):
            mask = masks[i] > 0.5 if masks[i].dtype != bool else masks[i]
            area = mask.sum()
            area_ratio = area / img_area
            if area_ratio < 0.005 or area_ratio > 0.70:
                continue

            if person_mask is not None:
                overlap = (mask & person_mask).sum()
                overlap_ratio = overlap / max(area, 1)
                person_area_ratio = area / max(person_mask.sum(), 1)
                if person_area_ratio < 0.01 or person_area_ratio > 0.60:
                    continue
                if overlap_ratio < 0.3:
                    continue
                person_ys, _ = np.where(person_mask)
                if len(person_ys) > 0:
                    person_top = person_ys.min()
                    person_bottom = person_ys.max()
                    person_height = person_bottom - person_top
                    mask_ys, _ = np.where(mask)
                    if len(mask_ys) > 0:
                        mask_center_y = mask_ys.mean()
                        upper_thresh = person_top + 0.15 * person_height
                        lower_thresh = person_top + 0.75 * person_height
                        if mask_center_y < upper_thresh or mask_center_y > lower_thresh:
                            continue
            kept_masks.append(mask)

        if not kept_masks:
            return None

        merged = np.zeros((img_h, img_w), dtype=bool)
        for m in kept_masks:
            merged = merged | m
        return merged


# ── Mask Operations ──────────────────────────────────────────────────
def dilate_mask(mask, person_height=None):
    if person_height is not None:
        radius = max(DILATION_PX_BASE, int(DILATION_RATIO * person_height))
    else:
        radius = DILATION_PX_BASE
    radius = max(1, radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    return dilated > 0


def blur_mask(mask, img_h, img_w):
    sigma = BLUR_SIGMA_RATIO * np.sqrt(img_h ** 2 + img_w ** 2)
    sigma = max(3.0, min(sigma, 80.0))
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), sigma)
    return np.clip(blurred, 0, 1)


# ── Per-folder processing ────────────────────────────────────────────
def process_folder(folder_path, segmentor):
    folder_name = folder_path.name
    parts = folder_name.split('_', 1)
    if len(parts) != 2:
        return False, f"cannot parse person_cloth from name"
    person_name, cloth_name = parts

    input_char_path = folder_path / "00_input_char_00001_.png"
    final_path = folder_path / "07_final_00001_.png"

    if not input_char_path.exists():
        return False, f"missing 00_input_char"
    if not final_path.exists():
        return False, f"missing 07_final"

    env_out_path = folder_path / final_path.name.replace("07_final", "08_env_color")
    cloth_out_path = folder_path / final_path.name.replace("07_final", "09_cloth_color")

    if env_out_path.exists() and cloth_out_path.exists():
        return True, "already done"

    # Load images
    try:
        img_p_pil = Image.open(input_char_path).convert("RGB")
        img_rfn_pil = Image.open(final_path).convert("RGB")
    except Exception as e:
        return False, f"load error: {e}"

    I_p = pil_to_np(img_p_pil)
    I_rfn = pil_to_np(img_rfn_pil)

    # Compute upper clothes mask for 07_final (needed for both steps)
    person_mask_rfn = segmentor.get_person_mask(img_rfn_pil)
    cloth_mask_rfn = segmentor.get_upper_clothes_mask(img_rfn_pil, person_mask_rfn)
    if cloth_mask_rfn is None:
        cloth_mask_rfn = np.zeros((I_rfn.shape[0], I_rfn.shape[1]), dtype=bool)

    # ── Step 1: Non-garment color correction ─────────────────────────
    if not env_out_path.exists():
        person_mask_p = segmentor.get_person_mask(img_p_pil)
        cloth_mask_p = segmentor.get_upper_clothes_mask(img_p_pil, person_mask_p)
        if cloth_mask_p is None:
            cloth_mask_p = np.zeros((I_p.shape[0], I_p.shape[1]), dtype=bool)

        bar_cloth_p = dilate_mask(cloth_mask_p, I_p.shape[0])
        bar_cloth_rfn = dilate_mask(cloth_mask_rfn, I_rfn.shape[0])

        hat_I_ng = mkl_transfer(
            I_rfn, I_p,
            src_mask=(~bar_cloth_rfn),
            ref_mask=(~bar_cloth_p),
            strength=LAMBDA_NG,
        )

        A = blur_mask(bar_cloth_rfn, I_rfn.shape[0], I_rfn.shape[1])
        A3 = np.repeat(A[:, :, np.newaxis], 3, axis=2)

        I_ng = ((1.0 - A3) * hat_I_ng.astype(np.float32) +
                A3 * I_rfn.astype(np.float32))
        I_ng = np.clip(I_ng, 0, 255).astype(np.uint8)
        Image.fromarray(I_ng).save(env_out_path)
    else:
        I_ng = pil_to_np(Image.open(env_out_path).convert("RGB"))

    # ── Step 2: Garment-level color correction ───────────────────────
    if cloth_out_path.exists():
        return True, "ok"

    pure_cloth_path = PURE_CLOTH_DIR / f"{cloth_name}.png"
    if not pure_cloth_path.exists():
        return True, f"pure cloth not found: {cloth_name}.png"

    try:
        I_g_pil = Image.open(pure_cloth_path)
        if I_g_pil.mode == 'RGBA':
            I_g_rgb = pil_to_np(I_g_pil.convert("RGB"))
            I_g_alpha = np.array(I_g_pil.split()[-1])
            M_cloth_g = I_g_alpha > 128
        else:
            I_g_rgb = pil_to_np(I_g_pil.convert("RGB"))
            M_cloth_g = np.ones((I_g_rgb.shape[0], I_g_rgb.shape[1]), dtype=bool)
    except Exception as e:
        return False, f"pure cloth load error: {e}"

    bar_cloth_g = dilate_mask(M_cloth_g)
    bar_cloth_ng = dilate_mask(cloth_mask_rfn, I_ng.shape[0])

    hat_I_gar = mkl_transfer(
        I_ng, I_g_rgb,
        src_mask=bar_cloth_ng,
        ref_mask=bar_cloth_g,
        strength=LAMBDA_G,
    )

    A = blur_mask(bar_cloth_ng, I_ng.shape[0], I_ng.shape[1])
    A3 = np.repeat(A[:, :, np.newaxis], 3, axis=2)

    I_gar = (A3 * hat_I_gar.astype(np.float32) +
             (1.0 - A3) * I_ng.astype(np.float32))
    I_gar = np.clip(I_gar, 0, 255).astype(np.uint8)
    Image.fromarray(I_gar).save(cloth_out_path)

    return True, "ok"


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--step1-only', action='store_true')
    parser.add_argument('--step2-only', action='store_true')
    args = parser.parse_args()

    import torch
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    segmentor = SAM3Segmentation(device=device)

    folders = sorted([
        d for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    if args.end < 0:
        args.end = len(folders)
    folders = folders[args.start:args.end]

    print(f"Processing {len(folders)} folders [{args.start}:{args.end}]")
    success = 0
    for folder in tqdm(folders, desc="Folders"):
        try:
            ok, msg = process_folder(folder, segmentor)
            if ok:
                success += 1
            else:
                tqdm.write(f"  [{folder.name}] FAIL: {msg}")
        except Exception as e:
            tqdm.write(f"  [{folder.name}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"Done. {success}/{len(folders)} succeeded.")


if __name__ == '__main__':
    main()
