#!/usr/bin/env python3
"""
Virtual Try-On Evaluation Pipeline
Non-pairwise metrics: Masked SSIM/LPIPS/PSNR, CLIP, VLM absolute scoring.

Usage:
    python eval_pipeline.py --gpu 0 --start 0 --end 100
"""

import os, sys, json, csv, argparse, pickle, hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import cv2

# ── Paths ───────────────────────────────────────────────────────────
SAM3_PATH = "/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI/custom_nodes/ComfyUI-Easy-Sam3"
sys.path.insert(0, SAM3_PATH)

SAM3_CKPT = "/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI/models/sam3/sam3.1_multiplex_fp16.safetensors"
VLM_PATH = "/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI/models/LLM/Qwen-VL/Qwen3-VL-4B-Instruct"
DATASET_CSV = "/hpc2hdd/home/dsaa2012_017/dl_test/data/dataset.csv"
OUTPUT_DIR = Path("/hpc2hdd/home/dsaa2012_017/dl_test/eval_results")
CACHE_DIR = OUTPUT_DIR / "cache"
MASKS_DIR = OUTPUT_DIR / "masks"
ALIGNED_DIR = OUTPUT_DIR / "aligned"
CROPS_DIR = OUTPUT_DIR / "crops"
METRICS_DIR = OUTPUT_DIR / "metrics"

for d in [MASKS_DIR, ALIGNED_DIR, CROPS_DIR, METRICS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────
CANVAS_WIDTH = 768
CANVAS_HEIGHT = 1024
TARGET_PERSON_HEIGHT_RATIO = 0.86
TARGET_CENTER_X_RATIO = 0.50
TARGET_CENTER_Y_RATIO = 0.52
PAD_VALUE = 127
SAM_CONFIDENCE_THRESHOLD = 0.3
DILATION_PX_BASE = 8
DILATION_RATIO = 0.015

# ── Helpers ─────────────────────────────────────────────────────────
def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)

def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def image_path_hash(path: str, prompt: str) -> str:
    h = hashlib.md5(f"{path}_{prompt}".encode()).hexdigest()[:16]
    return h

def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    # Convert to PNG in memory to avoid format issues with VLM processors
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize binary mask to target size."""
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    m = F.interpolate(m, size=(target_h, target_w), mode='nearest').squeeze().numpy()
    return m > 0.5

# ── SAM3 Segmentation ───────────────────────────────────────────────
class SAM3Segmentation:
    def __init__(self, device='cuda'):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print("Loading SAM3 model...")
        self.model = build_sam3_image_model(
            device=device,
            load_from_HF=False,
            checkpoint_path=SAM3_CKPT,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        # Ensure all parameters are float32 on the correct device
        self.model = self.model.float()
        # Re-move to device to catch any stragglers
        if device.startswith('cuda'):
            self.model = self.model.cuda()
        
        # Verify device placement
        param_devices = set()
        for name, p in self.model.named_parameters():
            param_devices.add(str(p.device))
        print(f"Model param devices: {param_devices}")
        
        self.processor = Sam3Processor(
            self.model, resolution=1008, device=device,
            confidence_threshold=SAM_CONFIDENCE_THRESHOLD
        )
        self.device = device
        print("SAM3 model loaded.")

    @torch.no_grad()
    def segment(self, image: Image.Image, prompt: str) -> Dict:
        """
        Returns dict with 'masks' (N,H,W bool array), 'boxes' (N,4), 'scores' (N,).
        Use prompt like "person" or "upper clothes".
        """
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
        
        # Convert to numpy
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
            boxes = boxes.cpu().numpy() if boxes is not None else np.zeros((masks.shape[0], 4))
            scores = scores.cpu().numpy() if scores is not None else np.zeros(masks.shape[0])
        
        # masks shape: (N, 1, H, W) or (N, H, W)
        if masks.ndim == 4:
            masks = masks[:, 0]
        
        return {'masks': masks, 'boxes': boxes, 'scores': scores}

    def get_person_mask(self, image: Image.Image) -> Optional[np.ndarray]:
        """Get person mask. Filters using heuristics from instructions."""
        result = self.segment(image, "person")
        masks = result['masks']
        scores = result['scores']
        boxes = result['boxes']
        
        if len(masks) == 0:
            return None
        
        # Select largest mask that satisfies basic criteria
        img_h, img_w = image.size[::-1]
        img_area = img_h * img_w
        best_mask = None
        best_score = -1
        
        for i in range(len(masks)):
            mask = masks[i] > 0.5 if masks[i].dtype != bool else masks[i]
            area = mask.sum()
            area_ratio = area / img_area
            
            # Filter: person should be reasonably sized
            if area_ratio < 0.01 or area_ratio > 0.95:
                continue
            
            # Prioritize masks near image center
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            cy, cx = ys.mean(), xs.mean()
            center_dist = np.sqrt((cy/img_h - 0.5)**2 + (cx/img_w - 0.5)**2)
            
            score = area * (1.0 - center_dist) * float(scores[i])
            if score > best_score:
                best_score = score
                best_mask = mask
        
        if best_mask is None and len(masks) > 0:
            # Fallback: use mask with highest score
            idx = np.argmax(scores)
            best_mask = masks[idx] > 0.5 if masks[idx].dtype != bool else masks[idx]
        
        return best_mask

    def get_upper_clothes_mask(self, image: Image.Image, person_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Get upper clothes mask. Filters using heuristics from instructions."""
        result = self.segment(image, "upper clothes")
        masks = result['masks']
        scores = result['scores']
        boxes = result['boxes']
        
        if len(masks) == 0:
            return None
        
        img_h, img_w = image.size[::-1]
        img_area = img_h * img_w
        
        kept_masks = []
        for i in range(len(masks)):
            mask = masks[i] > 0.5 if masks[i].dtype != bool else masks[i]
            area = mask.sum()
            area_ratio = area / img_area
            
            # Filter very small or very large
            if area_ratio < 0.005 or area_ratio > 0.70:
                continue
            
            # If we have person mask, check overlap
            if person_mask is not None:
                overlap = (mask & person_mask).sum()
                overlap_ratio = overlap / max(area, 1)
                person_area_ratio = area / max(person_mask.sum(), 1)
                
                if person_area_ratio < 0.01 or person_area_ratio > 0.60:
                    continue
                if overlap_ratio < 0.3:
                    continue
                
                # Check mask center is in upper body region
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
        
        # Merge all kept masks
        merged = np.zeros((img_h, img_w), dtype=bool)
        for m in kept_masks:
            merged = merged | m
        return merged

# ── Canonical Alignment ─────────────────────────────────────────────
def compute_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Compute bounding box from binary mask."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def compute_alignment_transform(person_bbox: Tuple[int, int, int, int],
                                 orig_w: int, orig_h: int) -> Tuple[float, float, float]:
    """
    Compute alignment parameters.
    Returns (scale, src_cx, src_cy) where:
      scale = target_person_height / person_bbox_height
      src_cx, src_cy = center of person bbox in original image
    """
    x_min, y_min, x_max, y_max = person_bbox
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    h_person = y_max - y_min
    target_h = TARGET_PERSON_HEIGHT_RATIO * CANVAS_HEIGHT
    scale = target_h / max(h_person, 1)
    return scale, cx, cy

def apply_alignment(image: np.ndarray, scale: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply canonical alignment to an image or mask.
    Returns (aligned_array, valid_mask).
    aligned_array: shape (CANVAS_HEIGHT, CANVAS_WIDTH, [C])
    valid_mask: shape (CANVAS_HEIGHT, CANVAS_WIDTH), True where original content exists
    """
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    is_mask = (image.ndim == 2) or (image.ndim == 3 and image.shape[2] == 1)
    
    if is_mask:
        resized = cv2.resize(image.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=resized.dtype)
        valid = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=bool)
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_value = PAD_VALUE if image.ndim == 3 else 0
        if image.ndim == 3:
            canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, image.shape[2]), pad_value, dtype=image.dtype)
        else:
            canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH), pad_value, dtype=image.dtype)
        valid = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=bool)
    
    # Target position on canvas
    target_cx = TARGET_CENTER_X_RATIO * CANVAS_WIDTH
    target_cy = TARGET_CENTER_Y_RATIO * CANVAS_HEIGHT
    
    # Source center after scaling
    scaled_cx = cx * scale
    scaled_cy = cy * scale
    
    # Paste offset
    paste_x = int(target_cx - scaled_cx)
    paste_y = int(target_cy - scaled_cy)
    
    # Compute overlap region
    src_x1 = max(0, -paste_x)
    src_y1 = max(0, -paste_y)
    src_x2 = min(new_w, CANVAS_WIDTH - paste_x)
    src_y2 = min(new_h, CANVAS_HEIGHT - paste_y)
    
    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if is_mask:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]
        else:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]
        valid[dst_y1:dst_y2, dst_x1:dst_x2] = True
    
    return canvas, valid

def align_image_and_masks(image: np.ndarray, person_mask: np.ndarray, 
                          cloth_mask: Optional[np.ndarray] = None) -> Dict:
    """
    Align image and masks to canonical canvas.
    Returns dict with aligned arrays and valid mask.
    """
    person_bbox = compute_bbox_from_mask(person_mask)
    h, w = image.shape[:2]
    
    if person_bbox is None:
        # Fallback: center crop/pad
        scale = min(CANVAS_WIDTH / w, CANVAS_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), PAD_VALUE, dtype=np.uint8)
        valid = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=bool)
        paste_x = (CANVAS_WIDTH - new_w) // 2
        paste_y = (CANVAS_HEIGHT - new_h) // 2
        canvas[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized
        valid[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = True
        
        p_resized = cv2.resize(person_mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        p_canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=bool)
        p_canvas[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = p_resized > 0.5
        
        result = {
            'aligned_image': canvas,
            'aligned_person_mask': p_canvas,
            'valid_mask': valid,
            'alignment_status': 'fallback_no_person_detected'
        }
    else:
        scale, cx, cy = compute_alignment_transform(person_bbox, w, h)
        aligned_img, valid = apply_alignment(image, scale, cx, cy)
        aligned_pmask, _ = apply_alignment(person_mask.astype(np.uint8), scale, cx, cy)
        aligned_pmask = aligned_pmask > 0.5
        status = 'ok'
        
        result = {
            'aligned_image': aligned_img,
            'aligned_person_mask': aligned_pmask,
            'valid_mask': valid,
            'alignment_status': status,
            'scale': scale,
            'src_center': (cx, cy),
            'src_bbox': person_bbox,
        }
    
    if cloth_mask is not None:
        if person_bbox is not None:
            aligned_cmask, _ = apply_alignment(cloth_mask.astype(np.uint8), scale, cx, cy)
            result['aligned_cloth_mask'] = aligned_cmask > 0.5
        else:
            c_resized = cv2.resize(cloth_mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            c_canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=bool)
            c_canvas[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = c_resized > 0.5
            result['aligned_cloth_mask'] = c_canvas
    
    return result

# ── Mask Generation ─────────────────────────────────────────────────
def compute_union_cloth_mask(cloth_a: np.ndarray, cloth_b: np.ndarray, 
                              person_height: int) -> np.ndarray:
    """Compute union cloth mask with dilation."""
    union = cloth_a | cloth_b
    dilation_radius = max(DILATION_PX_BASE, int(DILATION_RATIO * person_height))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_radius+1, 2*dilation_radius+1))
    dilated = cv2.dilate(union.astype(np.uint8), kernel)
    return dilated > 0

def compute_eval_masks(valid_a: np.ndarray, valid_b: np.ndarray,
                       person_a: np.ndarray, person_b: np.ndarray,
                       cloth_eval: np.ndarray) -> Dict:
    """Compute evaluation masks."""
    valid_both = valid_a & valid_b
    
    # Person union with small dilation
    person_union = person_a | person_b
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    person_union_dilated = cv2.dilate(person_union.astype(np.uint8), small_kernel) > 0
    
    # Person non-cloth eval mask
    eval_person_noncloth = valid_both & person_union_dilated & (~cloth_eval)
    
    # Full non-cloth eval mask
    eval_full_noncloth = valid_both & (~cloth_eval)
    
    return {
        'person_union': person_union,
        'cloth_union_eval': cloth_eval,
        'eval_mask_person_noncloth': eval_person_noncloth,
        'eval_mask_full_noncloth': eval_full_noncloth,
    }

# ── Preservation Metrics ────────────────────────────────────────────
def compute_masked_ssim(img_a: np.ndarray, img_b: np.ndarray, 
                        eval_mask: np.ndarray) -> Optional[float]:
    """Compute SSIM over masked region."""
    from skimage.metrics import structural_similarity as ssim
    if eval_mask.sum() < 100:
        return None
    # Crop to mask bounding box for efficiency
    ys, xs = np.where(eval_mask)
    if len(ys) < 100:
        return None
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    
    a_crop = img_a[y1:y2, x1:x2]
    b_crop = img_b[y1:y2, x1:x2]
    m_crop = eval_mask[y1:y2, x1:x2]
    
    # Compute SSIM on the crop and average over mask
    win_size_crop = min(11, min(y2-y1, x2-x1))
    if win_size_crop < 3:
        return None
    # Ensure odd window size
    if win_size_crop % 2 == 0:
        win_size_crop -= 1
    
    try:
        ssim_full, ssim_map = ssim(a_crop, b_crop, full=True, channel_axis=2, 
                                   data_range=255, win_size=win_size_crop)
    except Exception:
        return None
    if ssim_map is None:
        return float(ssim_full)
    
    # Average SSIM over mask (resize mask if needed)
    if ssim_map.shape != m_crop.shape:
        m_resized = cv2.resize(m_crop.astype(np.float32), 
                               (ssim_map.shape[1], ssim_map.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0.5
    else:
        m_resized = m_crop
    
    masked_mean = ssim_map[m_resized].mean()
    return float(masked_mean)

def compute_masked_psnr(img_a: np.ndarray, img_b: np.ndarray, 
                        eval_mask: np.ndarray) -> Optional[float]:
    """Compute PSNR over masked region."""
    if eval_mask.sum() < 100:
        return None
    diff = (img_a.astype(np.float32) - img_b.astype(np.float32))[eval_mask]
    mse = np.mean(diff ** 2)
    if mse < 1e-10:
        return 100.0
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    return float(psnr)

def compute_masked_lpips(lpips_model, img_a: np.ndarray, img_b: np.ndarray,
                         eval_mask: np.ndarray) -> Optional[float]:
    """Compute LPIPS over masked region using patch-based approach."""
    if eval_mask.sum() < 100:
        return None
    
    # Convert to tensors
    a_t = torch.from_numpy(img_a).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    b_t = torch.from_numpy(img_b).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Normalize to [-1, 1] for LPIPS
    a_t = a_t * 2 - 1
    b_t = b_t * 2 - 1
    
    with torch.no_grad():
        if torch.cuda.is_available():
            a_t = a_t.cuda()
            b_t = b_t.cuda()
        
        # Try spatial LPIPS first
        try:
            dist = lpips_model(a_t, b_t, spatial=True)
            lpips_map = dist.squeeze().cpu().numpy()
            
            if lpips_map.shape != eval_mask.shape:
                m_resized = cv2.resize(eval_mask.astype(np.float32),
                                      (lpips_map.shape[1], lpips_map.shape[0]),
                                      interpolation=cv2.INTER_NEAREST) > 0.5
            else:
                m_resized = eval_mask
            
            return float(lpips_map[m_resized].mean())
        except Exception:
            pass
        
        # Fallback: global LPIPS on masked image
        mask_t = torch.from_numpy(eval_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, size=a_t.shape[2:], mode='nearest')
        if torch.cuda.is_available():
            mask_t = mask_t.cuda()
        
        # Fill masked-out areas with gray
        gray = torch.zeros_like(a_t)
        a_masked = a_t * mask_t + gray * (1 - mask_t)
        b_masked = b_t * mask_t + gray * (1 - mask_t)
        
        dist = lpips_model(a_masked, b_masked)
        return float(dist.item())

# ── Garment Fidelity: CLIP ──────────────────────────────────────────
def compute_clip_similarity(clip_model, clip_preprocess, img_c: Image.Image,
                             img_b_crop: Image.Image) -> Optional[float]:
    """Compute CLIP image-image cosine similarity."""
    with torch.no_grad():
        device = next(clip_model.parameters()).device
        c_t = clip_preprocess(img_c).unsqueeze(0).to(device)
        b_t = clip_preprocess(img_b_crop).unsqueeze(0).to(device)
        
        c_feat = clip_model.encode_image(c_t)
        b_feat = clip_model.encode_image(b_t)
        
        c_feat = F.normalize(c_feat, dim=-1)
        b_feat = F.normalize(b_feat, dim=-1)
        
        sim = (c_feat * b_feat).sum(dim=-1)
        return float(sim.item())

def get_garment_crop(image: np.ndarray, cloth_mask: np.ndarray, expand_ratio=0.15) -> np.ndarray:
    """Crop garment region from image using cloth mask."""
    ys, xs = np.where(cloth_mask)
    if len(ys) < 50:
        return image
    
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    exp_w = int(box_w * expand_ratio)
    exp_h = int(box_h * expand_ratio)
    
    cx1 = max(0, x1 - exp_w)
    cy1 = max(0, y1 - exp_h)
    cx2 = min(image.shape[1], x2 + exp_w)
    cy2 = min(image.shape[0], y2 + exp_h)
    
    crop = image[cy1:cy2, cx1:cx2].copy()
    
    # Fill outside-mask pixels with gray
    crop_mask = cloth_mask[cy1:cy2, cx1:cx2]
    if crop.ndim == 3:
        crop[~crop_mask] = [127, 127, 127]
    
    return crop

# ── VLM Scoring ─────────────────────────────────────────────────────
class VLMScorer:
    def __init__(self, device='cuda'):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        print("Loading VLM model...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            VLM_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(VLM_PATH, trust_remote_code=True)
        print("VLM model loaded.")
    
    @torch.no_grad()
    def score_garment_fidelity(self, garment_img: Image.Image, 
                                result_img: Image.Image) -> Dict:
        prompt = "You are evaluating a virtual try-on result. Compare the target garment (first image) with the generated try-on result (second image). Score each dimension from 1 (worst) to 5 (best): category_consistency: Does the clothing category match the target? color_consistency: Does the color match the target? pattern_logo_consistency: Are patterns, logos, prints preserved? shape_structure_consistency: Does sleeve length, collar, neckline match? overall_garment_fidelity: Overall garment transfer quality. Return JSON only: {\"category_consistency\": N, \"color_consistency\": N, \"pattern_logo_consistency\": N, \"shape_structure_consistency\": N, \"overall_garment_fidelity\": N, \"reason\": \"brief\"}"
        return self._call_vlm(prompt, [garment_img, result_img])
    
    @torch.no_grad()
    def score_human_preference(self, person_img: Image.Image, garment_img: Image.Image,
                                result_img: Image.Image) -> Dict:
        prompt = "You are evaluating a virtual try-on result. Given: 1) original person image, 2) target garment, 3) generated try-on result. Score from 1 (worst) to 5 (best): garment_fidelity: How well does the clothing match the target garment? (category, color, texture, logo, structure) character_fidelity: How well is the person preserved? (face, hair, body shape, pose, skin tone) harmony_realism: How natural and realistic is the result? (fit, lighting, boundaries, artifacts) overall_quality: Overall try-on quality. Return JSON only: {\"garment_fidelity\": N, \"character_fidelity\": N, \"harmony_realism\": N, \"overall_quality\": N, \"reason\": \"brief\"}"
        return self._call_vlm(prompt, [person_img, garment_img, result_img])
    
    def _call_vlm(self, prompt: str, images: List[Image.Image]) -> Dict:
        messages = [{"role": "user", "content": [
            *[{"type": "image", "image": img} for img in images],
            {"type": "text", "text": prompt}
        ]}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        # Move to device and cast to model dtype for consistency
        target_device = self.model.device
        target_dtype = self.model.dtype
        inputs = {
            k: v.to(device=target_device, dtype=target_dtype if isinstance(v, torch.Tensor) and v.is_floating_point() else None)
            if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        
        generated_ids = [oid[len(iid):] for oid, iid in zip(generated_ids, inputs['input_ids'])]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except (json.JSONDecodeError, ValueError):
            pass
        
        return {"error": "json_parse_failed", "raw_response": response[:300]}

# ── Main Pipeline ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--start', type=int, default=0, help='Start row index')
    parser.add_argument('--end', type=int, default=-1, help='End row index (-1 for all)')
    parser.add_argument('--chunk', type=int, default=-1, help='Chunk index (0-based). Overrides --start/--end.')
    parser.add_argument('--total-chunks', type=int, default=1, help='Total number of chunks')
    parser.add_argument('--skip-vlm', action='store_true', help='Skip VLM scoring')
    parser.add_argument('--skip-metrics', action='store_true', help='Skip metrics, only do segmentation + alignment')
    parser.add_argument('--vlm-only', action='store_true', help='Only do VLM scoring (requires precomputed aligned)')
    args = parser.parse_args()
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    rows = []
    with open(DATASET_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Apply start/end filtering first
    if args.end < 0:
        args.end = len(rows)
    rows = rows[args.start:args.end]
    
    # Then apply chunking on the filtered rows
    if args.chunk >= 0:
        chunk_size = (len(rows) + args.total_chunks - 1) // args.total_chunks
        cstart = args.chunk * chunk_size
        cend = min(cstart + chunk_size, len(rows))
        rows = rows[cstart:cend]
        chunk_tag = f"chunk{args.chunk}_{args.start}_{args.end}"
        print(f"Chunk {args.chunk}/{args.total_chunks}: rows {cstart}-{cend} ({len(rows)} total)")
    else:
        chunk_tag = f"{args.start}_{args.end}"
        print(f"Processing {len(rows)} rows (from {args.start} to {args.end})")
    
    # Initialize models
    if not args.vlm_only:
        segmentor = SAM3Segmentation(device=device)
        
        import lpips
        lpips_model = lpips.LPIPS(net='alex', spatial=False).to(device)
        lpips_model.eval()
        
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        clip_model = clip_model.to(device)
        clip_model.eval()
    
    if not args.skip_vlm:
        vlm_scorer = VLMScorer(device=device)
    
    # Cache for person/garment segmentation (same tag as above)
    person_cache_file = CACHE_DIR / f"person_seg_{chunk_tag}.pkl"
    cloth_cache_file = CACHE_DIR / f"cloth_seg_{chunk_tag}.pkl"
    align_cache_file = CACHE_DIR / f"align_{chunk_tag}.pkl"
    
    person_cache = {}
    cloth_cache = {}
    align_cache = {}
    
    if person_cache_file.exists() and not args.vlm_only:
        with open(person_cache_file, 'rb') as f:
            person_cache = pickle.load(f)
        print(f"Loaded person cache: {len(person_cache)} entries")
    if cloth_cache_file.exists() and not args.vlm_only:
        with open(cloth_cache_file, 'rb') as f:
            cloth_cache = pickle.load(f)
        print(f"Loaded cloth cache: {len(cloth_cache)} entries")
    if align_cache_file.exists() and not args.vlm_only:
        with open(align_cache_file, 'rb') as f:
            align_cache = pickle.load(f)
        print(f"Loaded align cache: {len(align_cache)} entries")
    
    # Results storage
    preservation_results = []
    garment_results = []
    vlm_garment_results = []
    vlm_preference_results = []
    invalid_cases = []
    
    # Group rows by sample for efficient processing
    rows_by_sample = defaultdict(list)
    for row in rows:
        rows_by_sample[row['sample_id']].append(row)
    
    # For VLM-only mode, cache loaded person/garment images
    vlm_img_cache = {}
    
    # Process each sample
    for sample_id, sample_rows in tqdm(list(rows_by_sample.items()), desc="Samples"):
        first_row = sample_rows[0]
        person_path = first_row['person_image_path']
        garment_path = first_row['garment_image_path']
        person_name = first_row['person_name']
        cloth_name = first_row['cloth_name']
        
        # Cache images for VLM scoring
        if not args.skip_vlm:
            if person_path not in vlm_img_cache:
                vlm_img_cache[person_path] = load_image(person_path)
            if garment_path not in vlm_img_cache:
                vlm_img_cache[garment_path] = load_image(garment_path)
            img_a_pil_cached = vlm_img_cache[person_path]
            img_c_pil_cached = vlm_img_cache[garment_path]
        
        # Process person image A (shared across all methods for this sample)
        if not args.vlm_only:
            # Person segmentation for A
            a_cache_key = f"A:{person_path}"
            if a_cache_key in person_cache:
                a_person_mask_np = person_cache[a_cache_key]
            else:
                try:
                    img_a = load_image(person_path)
                    a_person_mask = segmentor.get_person_mask(img_a)
                    if a_person_mask is None:
                        invalid_cases.append({'sample_id': sample_id, 'model': 'N/A', 
                                            'reason': 'person_detection_failed_A'})
                        # Use fallback
                        a_person_mask = np.ones((img_a.size[1], img_a.size[0]), dtype=bool)
                    a_person_mask_np = a_person_mask
                    person_cache[a_cache_key] = a_person_mask_np
                except Exception as e:
                    import traceback
                    print(f"Error on A person seg {person_path}: {e}")
                    traceback.print_exc()
                    continue
            
            # Upper clothes for A
            a_cloth_cache_key = f"A_cloth:{person_path}"
            if a_cloth_cache_key in cloth_cache:
                a_cloth_mask_np = cloth_cache[a_cloth_cache_key]
            else:
                try:
                    img_a = load_image(person_path)
                    a_cloth_mask = segmentor.get_upper_clothes_mask(img_a, a_person_mask_np)
                    if a_cloth_mask is None:
                        a_cloth_mask = np.zeros((img_a.size[1], img_a.size[0]), dtype=bool)
                    a_cloth_mask_np = a_cloth_mask
                    cloth_cache[a_cloth_cache_key] = a_cloth_mask_np
                except Exception as e:
                    a_cloth_mask_np = np.zeros((img_a.size[1], img_a.size[0]), dtype=bool)
            
            # Align A
            a_align_key = f"A_align:{person_path}"
            if a_align_key in align_cache:
                a_aligned = align_cache[a_align_key]
            else:
                try:
                    img_a_np = pil_to_np(load_image(person_path))
                    a_aligned = align_image_and_masks(img_a_np, a_person_mask_np, a_cloth_mask_np)
                    align_cache[a_align_key] = a_aligned
                except Exception as e:
                    print(f"Error aligning A {person_path}: {e}")
                    continue
        
        # Process garment image C (shared)
        if not args.vlm_only:
            c_img = load_image(garment_path)
        
        # Process each method's output B
        for row in sample_rows:
            method = row['model']
            b_path = row['model_output_path']
            
            if not args.vlm_only and not args.skip_metrics:
                # 1. Person segmentation for B
                b_person_cache_key = f"B:{b_path}"
                if b_person_cache_key in person_cache:
                    b_person_mask_np = person_cache[b_person_cache_key]
                else:
                    try:
                        img_b = load_image(b_path)
                        b_person_mask = segmentor.get_person_mask(img_b)
                        if b_person_mask is None:
                            invalid_cases.append({'sample_id': sample_id, 'model': method,
                                                'reason': 'person_detection_failed_B'})
                            b_person_mask = np.ones((img_b.size[1], img_b.size[0]), dtype=bool)
                        b_person_mask_np = b_person_mask
                        person_cache[b_person_cache_key] = b_person_mask_np
                    except Exception as e:
                        print(f"Error on B person seg {b_path}: {e}")
                        continue
                
                # 2. Upper clothes for B
                b_cloth_cache_key = f"B_cloth:{b_path}"
                if b_cloth_cache_key in cloth_cache:
                    b_cloth_mask_np = cloth_cache[b_cloth_cache_key]
                else:
                    try:
                        img_b = load_image(b_path)
                        b_cloth_mask = segmentor.get_upper_clothes_mask(img_b, b_person_mask_np)
                        if b_cloth_mask is None:
                            b_cloth_mask = np.zeros((img_b.size[1], img_b.size[0]), dtype=bool)
                        b_cloth_mask_np = b_cloth_mask
                        cloth_cache[b_cloth_cache_key] = b_cloth_mask_np
                    except Exception as e:
                        b_cloth_mask_np = np.zeros((img_b.size[1], img_b.size[0]), dtype=bool)
                
                # 3. Align B
                b_align_key = f"B_align:{b_path}"
                if b_align_key in align_cache:
                    b_aligned = align_cache[b_align_key]
                else:
                    try:
                        img_b_np = pil_to_np(load_image(b_path))
                        b_aligned = align_image_and_masks(img_b_np, b_person_mask_np, b_cloth_mask_np)
                        align_cache[b_align_key] = b_aligned
                    except Exception as e:
                        print(f"Error aligning B {b_path}: {e}")
                        continue
                
                # 4. Generate evaluation masks
                a_pmask_aligned = a_aligned.get('aligned_person_mask')
                b_pmask_aligned = b_aligned.get('aligned_person_mask')
                a_cmask_aligned = a_aligned.get('aligned_cloth_mask')
                b_cmask_aligned = b_aligned.get('aligned_cloth_mask')
                
                if a_pmask_aligned is None or b_pmask_aligned is None:
                    continue
                
                # Person height for dilation
                person_ys = np.where(a_pmask_aligned | b_pmask_aligned)[0]
                person_height = person_ys.max() - person_ys.min() if len(person_ys) > 0 else CANVAS_HEIGHT
                
                # Compute union cloth mask
                cloth_eval = compute_union_cloth_mask(
                    a_cmask_aligned if a_cmask_aligned is not None else np.zeros_like(a_pmask_aligned),
                    b_cmask_aligned if b_cmask_aligned is not None else np.zeros_like(b_pmask_aligned),
                    person_height
                )
                
                # Compute eval masks
                eval_masks = compute_eval_masks(
                    a_aligned['valid_mask'], b_aligned['valid_mask'],
                    a_pmask_aligned, b_pmask_aligned,
                    cloth_eval
                )
                
                # Check valid area
                canvas_area = CANVAS_WIDTH * CANVAS_HEIGHT
                eval_pn = eval_masks['eval_mask_person_noncloth']
                eval_fn = eval_masks['eval_mask_full_noncloth']
                
                valid_ratio_pn = eval_pn.sum() / canvas_area
                valid_ratio_fn = eval_fn.sum() / canvas_area
                
                a_orig_size = f"{a_person_mask_np.shape[1]}x{a_person_mask_np.shape[0]}"
                b_orig_size = f"{b_person_mask_np.shape[1]}x{b_person_mask_np.shape[0]}"
                
                base_record = {
                    'sample_id': sample_id,
                    'person_name': person_name,
                    'cloth_name': cloth_name,
                    'model': method,
                    'alignment_status': f"A:{a_aligned.get('alignment_status','?')} B:{b_aligned.get('alignment_status','?')}",
                    'A_original_size': a_orig_size,
                    'B_original_size': b_orig_size,
                    'valid_area_ratio_person_noncloth': round(valid_ratio_pn, 6),
                    'valid_area_ratio_full_noncloth': round(valid_ratio_fn, 6),
                }
                
                # 5. Compute preservation metrics
                img_a_aligned = a_aligned['aligned_image']
                img_b_aligned = b_aligned['aligned_image']
                
                if valid_ratio_pn >= 0.05:
                    ssim_pn = compute_masked_ssim(img_a_aligned, img_b_aligned, eval_pn)
                    psnr_pn = compute_masked_psnr(img_a_aligned, img_b_aligned, eval_pn)
                    lpips_pn = compute_masked_lpips(lpips_model, img_a_aligned, img_b_aligned, eval_pn)
                else:
                    ssim_pn = psnr_pn = lpips_pn = None
                
                if valid_ratio_fn >= 0.05:
                    ssim_fn = compute_masked_ssim(img_a_aligned, img_b_aligned, eval_fn)
                    psnr_fn = compute_masked_psnr(img_a_aligned, img_b_aligned, eval_fn)
                    lpips_fn = compute_masked_lpips(lpips_model, img_a_aligned, img_b_aligned, eval_fn)
                else:
                    ssim_fn = psnr_fn = lpips_fn = None
                
                # 6. CLIP garment similarity
                clip_sim = None
                if b_cmask_aligned is not None:
                    b_garment_crop = get_garment_crop(img_b_aligned, b_cmask_aligned)
                    b_garment_pil = np_to_pil(b_garment_crop)
                    clip_sim = compute_clip_similarity(clip_model, clip_preprocess, c_img, b_garment_pil)
                
                preservation_results.append({
                    **base_record,
                    'Masked_SSIM_person_noncloth': round(ssim_pn, 6) if ssim_pn is not None else None,
                    'Masked_PSNR_person_noncloth': round(psnr_pn, 4) if psnr_pn is not None else None,
                    'Masked_LPIPS_person_noncloth': round(lpips_pn, 6) if lpips_pn is not None else None,
                    'Masked_SSIM_full_noncloth': round(ssim_fn, 6) if ssim_fn is not None else None,
                    'Masked_PSNR_full_noncloth': round(psnr_fn, 4) if psnr_fn is not None else None,
                    'Masked_LPIPS_full_noncloth': round(lpips_fn, 6) if lpips_fn is not None else None,
                })
                
                garment_results.append({
                    **base_record,
                    'CLIP_Garment_Similarity': round(clip_sim, 6) if clip_sim is not None else None,
                })
            
            # 7. VLM scoring
            if not args.skip_vlm:
                try:
                    img_b_pil = load_image(b_path)
                    
                    # Garment fidelity
                    gf_score = vlm_scorer.score_garment_fidelity(img_c_pil_cached, img_b_pil)
                    vlm_garment_results.append({
                        'sample_id': sample_id, 'model': method,
                        'VLM_category_consistency': gf_score.get('category_consistency'),
                        'VLM_color_consistency': gf_score.get('color_consistency'),
                        'VLM_pattern_logo_consistency': gf_score.get('pattern_logo_consistency'),
                        'VLM_shape_structure_consistency': gf_score.get('shape_structure_consistency'),
                        'VLM_overall_garment_fidelity': gf_score.get('overall_garment_fidelity'),
                        'VLM_garment_fidelity_json': json.dumps(gf_score),
                    })
                    
                    # Human preference
                    hp_score = vlm_scorer.score_human_preference(img_a_pil_cached, img_c_pil_cached, img_b_pil)
                    vlm_preference_results.append({
                        'sample_id': sample_id, 'model': method,
                        'VLM_garment_fidelity': hp_score.get('garment_fidelity'),
                        'VLM_character_fidelity': hp_score.get('character_fidelity'),
                        'VLM_harmony_realism': hp_score.get('harmony_realism'),
                        'VLM_overall_quality': hp_score.get('overall_quality'),
                        'VLM_preference_json': json.dumps(hp_score),
                    })
                except Exception as e:
                    invalid_cases.append({'sample_id': sample_id, 'model': method, 
                                        'reason': f'vlm_error: {str(e)[:100]}'})
    
    # Save caches
    if not args.vlm_only:
        with open(person_cache_file, 'wb') as f:
            pickle.dump(person_cache, f)
        with open(cloth_cache_file, 'wb') as f:
            pickle.dump(cloth_cache, f)
        with open(align_cache_file, 'wb') as f:
            pickle.dump(align_cache, f)
    
    # Write results
    if preservation_results:
        _write_csv(METRICS_DIR / f"preservation_metrics_{chunk_tag}.csv", 
                   preservation_results)
    if garment_results:
        _write_csv(METRICS_DIR / f"garment_fidelity_metrics_{chunk_tag}.csv",
                   garment_results)
    if vlm_garment_results:
        _write_csv(METRICS_DIR / f"vlm_garment_scores_{chunk_tag}.csv",
                   vlm_garment_results)
    if vlm_preference_results:
        _write_csv(METRICS_DIR / f"vlm_preference_scores_{chunk_tag}.csv",
                   vlm_preference_results)
    if invalid_cases:
        _write_csv(METRICS_DIR / f"invalid_cases_{chunk_tag}.csv",
                   invalid_cases)
    
    print("Done!")
    print(f"Preservation metrics: {len(preservation_results)}")
    print(f"Garment metrics: {len(garment_results)}")
    print(f"VLM garment scores: {len(vlm_garment_results)}")
    print(f"VLM preference scores: {len(vlm_preference_results)}")
    print(f"Invalid cases: {len(invalid_cases)}")

def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    main()
