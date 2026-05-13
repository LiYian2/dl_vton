import random
import os
import glob
import json
import time

# ── 从环境变量读取 batch 参数 ──
_ARGS_PORT = int(os.environ.get("COMFY_PORT", "8188"))
_ARGS_PERSON_IDX = int(os.environ.get("PERSON_IDX", "0"))
_ARGS_CLOTH_START = int(os.environ.get("CLOTH_START", "0"))
_ARGS_CLOTH_END = int(os.environ.get("CLOTH_END", "15"))

from comfy_script.runtime import *
load(f"http://127.0.0.1:{_ARGS_PORT}/")
from comfy_script.runtime.nodes import *
from pathlib import Path

# ============================================================
#  配置
# ============================================================
COMFYUI_ROOT = "/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI"
COMFYUI_INPUT = os.path.join(COMFYUI_ROOT, "input")
COMFYUI_OUTPUT = os.path.join(COMFYUI_ROOT, "output")

STAGE2_PROMPT_QWEN = """
            You are a garment-description assistant for a reference-conditioned virtual try-on pipeline.
            Analyze the target upper-body garment image. Describe only garment attributes needed for faithful transfer. Do not describe the person/model wearing it unless necessary to infer garment shape.
            Output valid JSON only. Use English only. Keep the entire output under 160 words. Use short phrases, not long sentences. Do not repeat information across fields. Do not describe transparency information. Do not invent unseen details. If a field is not applicable, use "".
            Return exactly this JSON:
            {
            "category_shape": "Max 20 words. Garment category, silhouette, fit, length, and overall volume.",
            "neckline_sleeves_coverage": "Max 20 words. Neckline/collar, sleeve length, shoulder/arm/chest/waist coverage or exposure.",
            "material_texture": "Max 10 words. Fabric impression, thickness, shine, knit, mesh, lace, leather, cotton, etc.",
            "colors_patterns_text": "Max 20 words. Main colors, secondary colors, patterns, prints, logos, and readable text.",
            "special_details": "Max 10 words. Buttons, zipper, seams, pockets, belt, lace, embroidery, asymmetry, decorations.",
            "case_prompt_for_flux": "Max 40 words. Compact garment-transfer clause for Flux2 preserving category, shape, neckline, sleeves, coverage, material, colors, pattern, logo/text, and special details."
            }

            Rules:
            - If text or logo is visible but unreadable, say "unreadable visible text/logo"; do not guess it.
            - If the garment is transparent, mesh, lace, sleeveless, off-shoulder, halter, low-neck, cropped, or unusually cut, explicitly describe exposed skin regions.
            - Focus on what must be preserved during try-on.
            """

STAGE2_PROMPT_FLUX2 = """TRYON. Dress the person in the target upper-body garment. 
    This is a garment layering task, not a body replacement task. Keep the person intact underneath the garment. The garment must look like real clothing worn on the body, naturally wrapping over the shoulders, chest, and upper torso.
    Respect the person's facing direction: if the person is shown from the back, do not generate front-side clothing details (e.g., front neckline, chest logo, buttons). The garment must match the visible back view only.
    Any body region that should be covered by the target garment must be covered by cloth, not left as visible skin. Do not expose nipples or uncovered chest inside the garment-covered area. 
    Edit only the target upper-garment region and leave all non-garment regions unchanged, preserving the original face, hair, pose, hands, fingers, arms, lower body, background, lighting, framing, and overall image style exactly as in the input. Do not change the lower body regions, do not undress the original trousers or dress. Please cover the nipples. Please do not make the nipples visible. Please erase the nipples.
    Follow the target garment faithfully: category, silhouette, neckline, sleeves, length, material, colors, logo, text, and details. Do not change garment design, erase the torso, do not replace the body with clothing, do not reveal hidden body parts, do not zoom out. Do not add/remove fingers, distort arms, or create fake shadows."""

STAGE3_PROMPT_QWEN = """
            You are a prompt-planning assistant for the refinement stage of a reference-conditioned virtual try-on pipeline.
            Input:
            1. Current try-on result image only.
            Your task is to generate compact case-specific constraints for a minimal refinement step. Focus only on visible information in the current image. The goal is to improve local coherence and realism without changing structure.
            Do not identify the person by name. Do not infer hidden body parts, hidden garment parts, or unseen design details. Do not suggest changing identity, pose, crop, background, style, or garment design.
            Output valid JSON only. Use English only. Keep the entire output under 130 words. Use short phrases, not long sentences. Do not repeat information across fields. If a field is not applicable, use "".
            Return exactly this JSON:
            {
            "preserve_global": "Max 28 words. Identity, face, hairstyle, pose, hands/fingers, crop, visible body range, background, and style to preserve.",
            "preserve_visible_garment": "Max 26 words. Visible garment category, silhouette, neckline, sleeves, material, colors, pattern, logo/text, and details to preserve.",
            "refine_targets": "Max 30 words. Local issues to improve: boundary blending, garment-body integration, lighting, skin tone, texture, wrinkles, seams, shadows, minor artifacts.",
            "case_prompt_for_flux": "Max 50 words. A direct Flux2 clause for subtle refinement with minimal structural change."
            }
            Rules:
            - Focus on local correction, not re-generation.
            - If hands or fingers are visible, explicitly preserve the exact visible finger count and arrangement.
            - If logo or text is visible and readable, preserve it exactly. If visible but unreadable, say 'unreadable visible text/logo'.
            - If some body parts or garment regions are hidden, cropped, shadowed, or ambiguous, keep them unchanged.
            - If no obvious artifact is visible, say refinement should be very subtle.`
        """

STAGE3_PROMPT_FLUX2 = """Primary task: remove local visual artifacts from the current try-on result without changing the overall image. This is a minimal refinement step, not a re-generation step. Also remove any nipple exposure if present.
            Clean up abnormal generated artifacts, mask remnants, floating or detached fabric pieces, unnatural black bands or patches, hard cut edges, broken garment boundaries, artificial arm/hand shadows, skin-color mismatches, and non-anatomical fragments. These abnormal artifacts must not remain in the refined image.
            Where the artifact overlaps an exposed skin region, replace it with natural continuous skin that matches the surrounding visible skin tone, lighting, texture, and shading. Maintain consistent skin color across the face, neck, shoulders, arms, hands, and all exposed skin regions. Do not create new body parts or reveal regions outside the original visible body range.
            Preserve the person's identity, face, hairstyle, expression, pose, body shape, hands, visible fingers, arms, tattoos, skin texture, camera angle, crop, framing, background, lighting, and visual style. Do not redraw the person, change the pose, change the framing, zoom out, or repaint the whole image.
            Preserve the garment category, silhouette, neckline, sleeves, length, material, color, texture, pattern, logo, text, transparency, and decorative details. Do not redesign the garment, hallucinate new details, alter logos/text, remove real tattoos, distort hands or arms, add/remove fingers, add arms, or over-smooth the face.
            Completion requirement: all abnormal local artifacts should be removed, while real tattoos, natural skin, garment details, identity, crop, background, and style remain unchanged."""

PERSON_FOLDER = "Designated Test_Character and clothes/Correspond"
CLOTH_FOLDER = "Designated Test_Character and clothes/Clothes"
NAKED_PERSON_FOLDER = "Designated Test_Character and clothes/NAKED"
OUTPUT_BASE = "final_24"
MANIFEST_FILE = "naked_manifest.json"


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


def process_vton_new(person_rel, cloth_rel, naked_rel, output_subdir, person_name, cloth_name, restore_naked=True):
    """新workflow (基于 prompt1.py)"""

    print(f"\n[VTON] {person_name} + {cloth_name}")
    print(f"  Naked: {naked_rel}")

    seed2 = random.randint(0, 2**32 - 1)
    seed3 = random.randint(0, 2**32 - 1)

    with Workflow():
        image, _ = LoadImage(person_rel)
        image3, _ = LoadImage(naked_rel)
        image5, _ = LoadImage(cloth_rel)
        input_char = image
        input_cloth = image5

        int_val = EasyInt(1280)
        image2, _, _, int_val, height = LayerUtilityImageScaleByAspectRatioV2(
            aspect_ratio='original', proportional_width=1, proportional_height=1,
            fit='crop', method='lanczos', round_to_multiple='16',
            scale_to_side='longest', scale_to_length=int_val,
            background_color='#000000', image=image
        )

        sam3_model = EasySam3ModelLoader(
            model='sam3.1_multiplex_fp16.safetensors',
            segmentor='image', device='cuda', precision='bf16'
        )
        masks, _, _, _, _ = EasySam3ImageSegmentation(
            sam3_model=sam3_model, images=image2, prompt='upper clothes',
            threshold=0.3, keep_model_loaded=False,
            add_background='none', detection_limit=-1
        )
        mask = INPAINTExpandMask(mask=masks, grow=5, blur=0, blur_type='gaussian')

        if restore_naked:
            image4 = ImageCompositeMasked(
                destination=image2, source=image3, x=0, y=0,
                resize_source=False, mask=mask
            )
        else:
            image4 = image3

        model = UNETLoader(
            'flux-2-klein/Flux2-Klein-9B-True-v2-bf16.safetensors', 'default'
        )
        model = LoraLoaderModelOnly(
            model=model,
            lora_name='F2K_9bb-一致性consist_20260225.safetensors',
            strength_model=0.8
        )

        text, _ = CRText(STAGE3_PROMPT_FLUX2)
        text2, _ = CRText(STAGE3_PROMPT_QWEN)
        text3, _ = CRText(STAGE2_PROMPT_FLUX2)
        text4, _ = CRText(STAGE2_PROMPT_QWEN)

        image6 = ImageScaleToTotalPixels(
            image=image5, upscale_method='lanczos', megapixels=1.2, resolution_steps=1
        )

        text4 = AILabQwenVL(
            model_name='Qwen3-VL-4B-Instruct',
            quantization='4-bit (VRAM-friendly)',
            attention_mode='auto',
            preset_prompt='🖼️ Detailed Description',
            custom_prompt=text4,
            max_tokens=1024,
            keep_model_loaded=274870569668538,
            seed=seed2,
            image=image6
        )
        text4 = LayerUtilityPurgeVRAMV2(anything=text4, purge_cache=True, purge_models=True)

        string, _ = CRTextConcatenate(text1=text3, text2=text4, separator='')
        string = EasyShowAnything(string)

        clip = CLIPLoader(
            clip_name='qwen_3_8b_fp8mixed.safetensors',
            type='flux2',
            device='default'
        )
        conditioning = CLIPTextEncode(string, clip)
        vae = VAELoader('flux2-vae.safetensors')

        latent = VAEEncode(image4, vae) #person latent
        conditioning2 = ReferenceLatent(conditioning, latent)
        latent2 = VAEEncode(image6, vae) #cloth latent
        conditioning2 = ReferenceLatent(conditioning2, latent2)
        conditioning3 = ConditioningZeroOut(conditioning)
        conditioning3 = ReferenceLatent(conditioning3, latent)
        conditioning3 = ReferenceLatent(conditioning3, latent2)

        guider = CFGGuider(model=model, positive=conditioning2, negative=conditioning3, cfg=1)
        sampler = KSamplerSelect('euler')
        sigmas = Flux2Scheduler(steps=5, width=int_val, height=height)
        latent3 = EmptyFlux2LatentImage(width=int_val, height=height, batch_size=1)

        noise2 = RandomNoise(seed2)
        latent3, _ = SamplerCustomAdvanced(
            noise=noise2, guider=guider, sampler=sampler,
            sigmas=sigmas, latent_image=latent3
        )
        latent3 = LayerUtilityPurgeVRAMV2(anything=latent3, purge_cache=True, purge_models=True)
        image7 = VAEDecode(latent3, vae)

        masks2, _, _, _, _ = EasySam3ImageSegmentation(
            sam3_model=sam3_model, images=image7, prompt='upper clothes',
            threshold=0.4, keep_model_loaded=False,
            add_background='none', detection_limit=-1
        )
        mask2 = AddMask(masks, masks2)
        mask2 = MaskFillHoles(masks=mask2)
        mask2 = INPAINTExpandMask(mask=mask2, grow=4, blur=2, blur_type='gaussian')

        image8 = ImageCompositeMasked(
            destination=image4, source=image7, x=0, y=0,
            resize_source=False, mask=mask2
        )


        text2 = AILabQwenVL(
            model_name='Qwen3-VL-4B-Instruct',
            quantization='4-bit (VRAM-friendly)',
            attention_mode='auto',
            preset_prompt='🖼️ Detailed Description',
            custom_prompt=text2,
            max_tokens=1024,
            keep_model_loaded=274870569668538,
            seed=seed3,
            image=image8
        )
        text2 = LayerUtilityPurgeVRAMV2(anything=text2, purge_cache=True, purge_models=True)

        string2, _ = CRTextConcatenate(text1=text, text2=text2, separator='')
        conditioning4 = CLIPTextEncode(string2, clip)

        int2_val = EasyInt(1600)
        image9, _, _, int2_val, height2 = LayerUtilityImageScaleByAspectRatioV2(
            aspect_ratio='original', proportional_width=1, proportional_height=1,
            fit='crop', method='lanczos', round_to_multiple='16',
            scale_to_side='longest', scale_to_length=int2_val,
            background_color='#000000', image=image8
        )

        latent4 = VAEEncode(image9, vae)
        conditioning5 = ReferenceLatent(conditioning4, latent4)
        conditioning6 = ReferenceLatent(conditioning4, latent4)

        guider2 = CFGGuider(model=model, positive=conditioning5, negative=conditioning6, cfg=1)
        sampler2 = KSamplerSelect('euler')
        sigmas2 = Flux2Scheduler(steps=5, width=int2_val, height=height2)
        latent5 = EmptyFlux2LatentImage(width=int2_val, height=height2, batch_size=1)

        noise3 = RandomNoise(seed3)
        latent5, _ = SamplerCustomAdvanced(
            noise=noise3, guider=guider2, sampler=sampler2,
            sigmas=sigmas2, latent_image=latent5
        )
        latent5 = LayerUtilityPurgeVRAMV2(anything=latent5, purge_cache=True, purge_models=True)
        image10 = VAEDecode(latent5, vae)
        image10 = ImageSmartSharpen(
            image=image10, noise_radius=7, preserve_edges=0.75, sharpen=4, ratio=1
        )

        image11 = EasyImageColorMatch(
            image_ref=image8, image_target=image10,
            method='wavelet', image_output='Preview', save_prefix='ComfyUI'
        )

        pfx = lambda name: f"{output_subdir}/{name}"

        SaveImage(input_char,  pfx("00_input_char"))
        SaveImage(input_cloth, pfx("01_input_cloth"))

        images = DrawMaskOnImage(image=image2, mask=masks, color='0, 255, 0', device='cpu')
        SaveImage(images, pfx("debug_upper_clothes_mask1"))
        SaveImage(image4, pfx("02_naked_char"))

        image12 = ImageCompositeMasked(
            destination=image2, source=image4, x=0, y=0,
            resize_source=False, mask=masks
        )
        SaveImage(image12, pfx("03_naked_char_restored"))
        SaveImage(image7, pfx("04_vton_raw"))

        images2 = DrawMaskOnImage(image=image4, mask=mask2, color='0, 255, 0', device='cpu')
        SaveImage(images2, pfx("05_union_mask_on_char"))
        SaveImage(image8, pfx("06_union_mask_restoration"))
        SaveImage(image10, pfx("07_final"))
        SaveImage(image11, pfx("final"))

    print(f"  Done: {person_name} + {cloth_name}")


def main():
    person_folder_abs = os.path.join(COMFYUI_INPUT, PERSON_FOLDER)
    cloth_folder_abs = os.path.join(COMFYUI_INPUT, CLOTH_FOLDER)
    naked_folder_abs = os.path.join(COMFYUI_INPUT, NAKED_PERSON_FOLDER)

    person_images = collect_images(person_folder_abs)
    cloth_images = collect_images(cloth_folder_abs)
    naked_images = collect_images(naked_folder_abs)

    person_images_rel = [os.path.relpath(p, COMFYUI_INPUT) for p in person_images]
    cloth_images_rel = [os.path.relpath(c, COMFYUI_INPUT) for c in cloth_images]
    naked_images_rel = [os.path.relpath(n, COMFYUI_INPUT) for n in naked_images]

    person_images_sorted = sorted(person_images_rel)
    cloth_images_sorted = sorted(cloth_images_rel, key=lambda c: os.path.basename(c))

    print(f"Port: {_ARGS_PORT}")
    print(f"Person Images Found: {len(person_images_sorted)}")
    print(f"Cloth Images Found: {len(cloth_images_sorted)}")
    print(f"Naked Images Found: {len(naked_images_rel)}")
    print(f"Batch: person_idx={_ARGS_PERSON_IDX}, cloth {_ARGS_CLOTH_START}-{_ARGS_CLOTH_END}")

    if _ARGS_PERSON_IDX >= len(person_images_sorted):
        print(f"ERROR: person_idx {_ARGS_PERSON_IDX} out of range (0-{len(person_images_sorted) - 1})")
        return

    person_rel = person_images_sorted[_ARGS_PERSON_IDX]
    p_path = Path(person_rel)
    p_name = p_path.stem

    naked_rel_path = str(p_path.with_name("n" + p_path.name))
    naked_rel_path = naked_rel_path.replace("jpg", "png").replace("webp", "png").replace("jpeg", "png")
    naked_rel_path = naked_rel_path.replace("Correspond", "NAKED")

    if naked_rel_path not in naked_images_rel:
        print(f"ERROR: No naked image for {person_rel}, expected {naked_rel_path}")
        return
    naked_rel = naked_rel_path

    cloth_start = max(0, _ARGS_CLOTH_START)
    cloth_end = min(len(cloth_images_sorted) - 1, _ARGS_CLOTH_END)
    if cloth_start > cloth_end:
        print(f"ERROR: cloth_start ({cloth_start}) > cloth_end ({cloth_end})")
        return

    batch_clothes = cloth_images_sorted[cloth_start:cloth_end + 1]
    total = len(batch_clothes)
    print(f"\nProcessing person '{p_name}' with {total} clothes (range {cloth_start}-{cloth_end})")
    print("=" * 60)

    success, fail, current = 0, 0, 0

    for cloth_rel in batch_clothes:
        current += 1
        cname = os.path.splitext(os.path.basename(cloth_rel))[0]
        output_subdir = f"{OUTPUT_BASE}/{p_name}_{cname}"

        try:
            process_vton_new(person_rel, cloth_rel, naked_rel,
                             output_subdir, p_name, cname, restore_naked=False)
            success += 1
        except Exception as e:
            fail += 1
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"  Progress: {current}/{total} | OK {success} FAIL {fail}")

    print(f"\n{'=' * 60}")
    print(f"COMPLETED: OK {success} FAIL {fail}")
    print(f"Output: {os.path.join(COMFYUI_OUTPUT, OUTPUT_BASE)}")


if __name__ == "__main__":
    main()
