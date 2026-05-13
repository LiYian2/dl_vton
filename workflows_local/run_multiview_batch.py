import random
import os
import glob
import json
import time

_ARGS_PORT = int(os.environ.get("COMFY_PORT", "8188"))
_ARGS_START_IDX = int(os.environ.get("START_IDX", "0"))
_ARGS_END_IDX = int(os.environ.get("END_IDX", "2"))

from comfy_script.runtime import *
load(f"http://127.0.0.1:{_ARGS_PORT}/")
from comfy_script.runtime.nodes import *
from pathlib import Path

COMFYUI_ROOT = "/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI"
COMFYUI_INPUT = os.path.join(COMFYUI_ROOT, "input")
COMFYUI_OUTPUT = os.path.join(COMFYUI_ROOT, "output")
MULTIVIEW_INPUT = "multiview_input"
CLOTH_FOLDER = "Designated Test_Character and clothes/Pure Clothes"
OUTPUT_BASE = "multiview_v4"

QWEN_PROMPT = """
你是一名电影级视觉导演 + 3D 多视角生成专家，精通多视角一致性、相机矩阵与光影物理。
【任务】
根据输入的单张人物写真图片以及单张衣服图片，生成 8 段多视角提示词，用于图像生成模型（如 SDXL + Multi-View ControlNet），输出同一人物、同一姿态、同一环境、同一光照下的不同相机视角图像。
目标是构建一组：
视角连续（环绕主体旋转）你是一名电影级视觉导演 + 3D 多视角生成专家，精通材质反推与物理光照。
【任务】
根据输入的两张人物写真图片：
图片 A：人物整体（半身或全身）写真，包含姿态、环境、整体光照
图片 B：同一人物的上半身衣服特写（纯衣服细节，可包含少量皮肤/颈部）
生成 8 段多视角提示词，用于图像生成模型（如 SDXL + Multi-View ControlNet），输出同一人物、同一姿态、同一套服装（精细材质一致）、同一环境、同一世界光照下的不同相机视角图像。
目标是构建一组：
视角连续（环绕主体旋转）
姿态固定（动作、表情、头发动态不变）
服装材质完全一致（纹理、图案、缝线、褶皱规律）
光线统一（世界坐标系中的光源方向固定）
风格稳定（影调、对比、色调一致）
的高端多视角写真序列。
【人物与服装一致性锁定（必须执行）】

从图片 A 锁定：

人脸（五官结构 / 脸型 / 气质）

发型（长度 / 卷度 / 分缝 / 发量 / 动态方向）

妆容（风格 / 浓淡 / 色系）

身体姿势（全身姿态、手部位置、视线方向）—— 严格固定，不得随视角改变

配饰（包 / 手机 / 饰品 / 鞋，如有）

从图片 B 锁定（覆盖图片 A 中衣服可能模糊或光照干扰的部分）：

服装款式（领型 / 肩型 / 袖长 / 腰线 / 下摆）

服装颜色（精确 RGB 倾向 / 色相 / 饱和度 / 明度）

服装材质（棉 / 丝 / 皮革 / 针织 / 牛仔 / 亮片 / 蕾丝等，及其表面反光特性）

纹理与图案（条纹 / 格纹 / 印花 / logo / 刺绣的位置、大小、重复规律）

剪裁细节（缝线颜色与走向、纽扣 / 拉链 / 口袋的形状与位置）

褶皱与垂坠规律（从特写图中推断布料软硬及自然悬垂形态）

严禁：

换脸 / 换发型 / 改妆容

改变姿势或视线世界方向

改变衣服的任何细节（包括图案走向、纽扣位置、材质反光）

引入图片 B 中的光照（图片 B 的光线仅用于看清纹理，最终光照以图片 A 为准）

【双图信息融合规则（隐式完成）】

环境特征：仅从图片 A 提取

空间类型（建议使用纯色无影墙 / 无缝背景纸 / 简单纹理墙面，避免视角矛盾）

地面材质（磨砂亚克力 / 木地板 / 水泥地）

背景元素（若有标志物，需明确其 3D 位置；否则统一为单色）

光线特征（世界坐标系固定）：仅从图片 A 推断

主光方向（绝对方向：前 / 左前 / 右侧 / 左后等）

补光方向与强度比

光质（柔光 / 硬光）

色温（开尔文值）

轮廓光 / 背景光是否存在

衣服外观：完全以图片 B 为基准，覆盖图片 A 中衣服可能因光照或分辨率产生的偏差

材质反光特性需配合图片 A 的光照重新计算（例如：皮革在左侧光下应有高光在左侧，右侧背光面暗）

图案在身体表面的透视映射需保持物理正确（条纹随身体曲面弯曲）

【多视角生成规则】

每一段必须包含：

view type（视角类型：方位角 + 俯仰角）

camera distance（相机距离：近 / 中 / 全，所有视角一致）

lens（焦段 + 光圈，所有视角一致，建议 85mm f/4）

subject orientation（主体朝向相对相机静止，相机环绕）

environment（环境在视角变化后的自然延续，背景为无影墙或简单场景）

lighting（相对于世界坐标的光线描述，阴影随视角变化）

depth of field（景深统一）

composition（构图统一，如居中全身或半身）

【视角递进结构（8个视角，水平环绕）】

按以下顺序，方位角递增，俯仰角固定 0°：

View 1：Front (0°) —— 姿态与图片 A 尽可能一致
View 2：Front-left 45°
View 3：Left 90°
View 4：Back-left 135°
View 5：Back 180°
View 6：Back-right 225°
View 7：Right 270°
View 8：Front-right 315°

【姿态与视线固定规则】

全身姿态完全固定：站立或静坐，双手位置、手指姿势、重心分布与图片 A 相同

表情固定：与图片 A 一致（中性 / 微笑 / 冷峻）

视线方向：固定为世界坐标系中的绝对方向

【光线连续性规则（核心）】

主光方向必须在世界坐标系中固定。从图片 A 反推绝对方向。

提示词中必须使用绝对方向描述。

补光、轮廓光方向同样绝对固定。

色温、光比、光质在所有视角中完全相同。

【镜头与环境统一】

所有视角使用相同焦段：85mm（半身），50mm（全身），光圈 f/2.8–f/5.6。

相机距离：使主体在画面中的大小恒定。

背景：建议替换为"纯色无影墙（cyclorama wall），墙角圆弧过渡，地面轻微反射"。

【输出格式】

仅输出提示词，不要解释，不要分析。每个视角之间空一行。

格式：

View 1:[view type], [camera distance], [lens: xx mm, f/x.x], [azimuth: X°, elevation: Y°], [subject pose: exactly as in image A], [gaze: absolute direction derived from image A], [garment: from image B details — fabric, pattern, color, stitching, buttons], [environment: e.g. cyclorama wall], [lighting: absolute key/fill/rim directions and quality], [dof: f/4 sharp figure], [composition: centered, full body]

View 2:...

...

View 8:...

禁止空行（指输出内容中段首不要空行，但视角之间空一行是允许的），禁止任何前言、后记或解释性废话和负面提示。

姿态固定（动作、表情、衣物不变）

光线统一（世界坐标系中的光源方向固定）

风格稳定（影调、对比、色调一致）

的高端多视角写真序列，而不是8张无关联图片。

【人物一致性锁定（必须执行）】
所有视角必须保持完全一致：

人脸（五官结构 / 脸型 / 气质）

发型（长度 / 卷度 / 分缝 / 发量 / 动态）

妆容（风格 / 浓淡 / 色系）

服装（款式 / 颜色 / 材质 / 剪裁 / 褶皱形态）

配饰（包 / 手机 / 饰品 / 鞋）

身体姿势（全身姿态、手部位置、视线方向）—— 严格固定，不得随视角改变

严禁：

换脸 / 换发型 / 换衣服 / 改风格

改变姿势或视线相对方向

【视觉信息反推（先隐式完成）】
从输入图像中提取并贯穿所有视角：

人物特征（同上）

环境特征：

空间类型（建议使用纯色无影墙 / 无缝背景纸 / 简单纹理墙面）

地面材质（如磨砂亚克力、木地板、水泥地）

背景元素（若有标志物，需明确其3D空间位置）

光线特征（世界坐标系固定）：

主光方向（绝对方向：前方 / 左侧 / 右侧 / 左侧后方等）

补光方向与强度

光质（柔光 / 硬光）

色温（开尔文值）

是否添加轮廓光（rim light）或背景光

【多视角生成规则（核心执行）】
每一段必须包含：

view type（视角类型：方位角 + 俯仰角）

camera distance（相机距离：近 / 中 / 全）

lens（焦段 + 光圈，所有视角保持一致）

subject orientation（主体朝向：相对相机的角度，但实际主体静止，相机运动）

environment（环境在视角变化后的自然延续）

lighting（相对于世界坐标的光线描述，注意阴影随视角变化）

depth of field（景深，所有视角统一）

composition（构图规则，如三分法、居中）

【视角递进结构（必须遵守）】
按环绕顺序排列，相机水平旋转一周（方位角从 0° 递增），俯仰角固定为 0°（平视）。

推荐顺序（8个视角）：
View 1：Front (0°) —— 与输入图片视角最接近
View 2：Front-left 45° (-45°)
View 3：Left 90° (-90°)
View 4：Back-left 135° (135°)
View 5：Back 180°
View 6：Back-right 225° (或 -135°)
View 7：Right 270° (或 -90°)
View 8：Front-right 315° (或 -45°)

【视角连续性规则】

主体姿态完全固定（站立或坐姿，双手位置不变，表情不变，视线方向保持绝对世界方向）

【光线连续性规则（核心）】
光源必须定义在世界坐标系中，而非跟随相机。

主光方向：从输入图像反推出绝对方向（例如：来自相机左侧 45° 上方）

所有视角的描述中，光线方向绝对不变，仅描述阴影和亮部相对于新视角的可见变化

色温、光质、光比在所有视角中完全相同

不允许出现"左侧光"这种相对表述，应写"世界坐标系中来自东偏北45°方向的光"

【镜头语言规范】
所有视角使用一致的镜头参数：

焦段：85mm（半身像）、50mm（全身）、或输入图像推断值

光圈：f/2.8 – f/5.6（保证足够景深，避免因视角变化导致虚化不一致）

相机距离：保持主体在画面中大小恒定

【环境与背景处理】

若输入图像背景是真实三维空间（如街道、房间），多视角下背景必须物理合理延伸。建议在提示词中强制替换为"纯色无影墙（cyclorama wall）"或"360° HDRI 环境贴图（例如中性灰棚拍环境）"

明确输出背景类型：如"干净的无影墙，墙角为圆弧过渡，地面反射均匀"

【输出格式（严格执行）】
仅输出提示词，不要解释，不要分析

每个视角之间必须空一行

格式如下：

View 1:[view type], [camera distance], [lens: xx mm, f/x.x], [azimuth: X°, elevation: Y°], [subject pose unchanged], [environment description], [lighting: absolute direction + quality], [depth of field], [composition]

View 2:...

...

View 8:...

【单视角模板示例】
View 1: Front view, full body, lens: 85mm, f/4.0, azimuth 0°, elevation 0°, standing relaxed with hands in coat pockets, gaze fixed straight ahead (world 0°), hair static, solid gray cyclorama wall with slight floor reflection, key light from world left-front 45° at 2m, 5600K softbox, fill from right rear at 1/2 power, dof: sharp figure, background slightly soft at f/4, centered composition, feet aligned with frame bottom
"""


def collect_multiview_images():
    input_base = os.path.join(COMFYUI_INPUT, MULTIVIEW_INPUT)
    images = []
    for subdir in sorted(os.listdir(input_base)):
        subdir_path = os.path.join(input_base, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for f in sorted(os.listdir(subdir_path)):
            if f.lower().endswith('.png'):
                abs_path = os.path.join(subdir_path, f)
                rel_path = os.path.relpath(abs_path, COMFYUI_INPUT)
                images.append(rel_path)
    return images


def find_cloth_rel(cloth_id):
    cloth_dir = os.path.join(COMFYUI_INPUT, CLOTH_FOLDER)
    for ext in ['', '.png', '.jpg', '.jpeg', '.webp']:
        candidate = os.path.join(cloth_dir, f"{cloth_id}{ext}")
        if os.path.exists(candidate):
            return os.path.relpath(candidate, COMFYUI_INPUT)
    return None


def process_multiview(person_rel, cloth_rel, output_subdir, person_name, view_idx, seed_qwen, seed_sampler):
    with Workflow():
        clip = CLIPLoader(clip_name='qwen_3_8b_fp8mixed.safetensors', type='flux2', device='default')
        # conditioning_neg = CLIPTextEncode('low quality, blurry, plastic skin', clip)

        image, _ = LoadImage(person_rel)
        image, _, _, width, height = LayerUtilityImageScaleByAspectRatioV2(
            aspect_ratio='original', proportional_width=1, proportional_height=1,
            fit='crop', method='lanczos', round_to_multiple='16',
            scale_to_side='longest', scale_to_length=1280,
            background_color='#000000', image=image
        )

        vae = VAELoader('flux2-vae.safetensors')
        latent = VAEEncode(image, vae)

        model = UNETLoader('flux-2-klein/Flux2-Klein-9B-True-v2-bf16.safetensors', 'default')
        model = LoraLoaderModelOnly(model=model, lora_name='F2K_9bb-一致性consist_20260225.safetensors', strength_model=0.5)
        model = ModelPassThrough(model)

        string = PrimitiveStringMultiline(
            'Edit image into the given view of the same person wearing the same garment. '
            'Preserve all visible garment identity cues from the original image, including '
            'the garment type, color palette, silhouette, sleeve length, collar/neckline '
            'structure as much as physically plausible, texture, and any visible patterns. '
            'Do not change the shoes of the character if present.'
        )
        string2_prompt = PrimitiveStringMultiline(QWEN_PROMPT)

        image2, _ = LoadImage(cloth_rel)

        string2 = QwenVLMultiImageAdvanced(
            images=image, model_name='Qwen/Qwen3-VL-4B-Instruct',
            system_prompt='You are a helpful assistant.',
            user_prompt=string2_prompt,
            quantization='4-bit (VRAM-friendly)',
            max_tokens=4096, temperature=0.7, top_p=0.9, top_k=50,
            repetition_penalty=1.1, num_beams=1,
            keep_model_loaded=True, seed=seed_qwen,
            device='auto', images_batch_2=image2
        )
        string2 = SomethingToString(input=string2, prefix='', suffix='')
        string2 = EasyCleanGpuUsed(string2)
        string2 = EasyShowAnything(string2)
        string2, _ = EasyPromptLine(prompt=string2, start_index=0, max_rows=1000, remove_empty_lines=True)

        string3, _ = CRTextConcatenate(text1=string, text2=string2, separator='')
        string4, _ = EasyPromptLine(prompt=string3, start_index=0, max_rows=1000, remove_empty_lines=True)

        conditioning2 = CLIPTextEncode(string4, clip)
        conditioning3 = ReferenceLatent(conditioning2, latent)

        image3, _, _, _, _ = LayerUtilityImageScaleByAspectRatioV2(
            aspect_ratio='original', proportional_width=1, proportional_height=1,
            fit='crop', method='lanczos', round_to_multiple='16',
            scale_to_side='longest', scale_to_length=1280,
            background_color='#000000', image=image2
        )
        latent2 = VAEEncode(image3, vae)
        conditioning3 = ReferenceLatent(conditioning3, latent2)

        conditioning4 = ConditioningZeroOut(conditioning2)
        conditioning4 = ReferenceLatent(conditioning4, latent)
        conditioning4 = ReferenceLatent(conditioning4, latent2)

        latent3 = EmptyFlux2LatentImage(width=width, height=height, batch_size=1)
        latent3 = KSampler(model=model, seed=seed_sampler, steps=10, cfg=1,
                           sampler_name='lcm', scheduler='normal',
                           positive=conditioning3, negative=conditioning4,
                           latent_image=latent3, denoise=1)

        image4 = VAEDecode(latent3, vae)
        image5 = ColorMatch(image_ref=image, image_target=image4, method='mkl', strength=0.1, multithread=True)
        image5 = ImageCASharpening(image5, 0)

        SaveImage(image5, f"{output_subdir}/{person_name}_")

    print(f"  Done: {person_name}")


def main():
    all_images = collect_multiview_images()
    print(f"Port: {_ARGS_PORT}")
    print(f"Multiview Images Found: {len(all_images)}")
    print(f"Batch: idx {_ARGS_START_IDX}-{_ARGS_END_IDX}")

    start = max(0, _ARGS_START_IDX)
    end = min(len(all_images) - 1, _ARGS_END_IDX)

    if start > end:
        print(f"ERROR: start ({start}) > end ({end})")
        return

    batch = all_images[start:end + 1]
    total = len(batch)
    print(f"Processing {total} images")
    print("=" * 60)

    success, fail = 0, 0

    for idx, person_rel in enumerate(batch):
        basename = os.path.splitext(os.path.basename(person_rel))[0]
        parts = basename.split('_')
        if len(parts) < 2:
            print(f"  SKIP {basename}: cannot parse cloth id")
            continue
        cloth_id = parts[-1]

        cloth_rel = find_cloth_rel(cloth_id)
        if cloth_rel is None:
            print(f"  SKIP {basename}: cloth {cloth_id} not found in Pure Clothes")
            fail += 1
            continue

        output_subdir = f"{OUTPUT_BASE}/{basename}"
        print(f"\n[{idx + 1}/{total}] {basename}  cloth={cloth_id}")

        try:
            seed_qwen = random.randint(0, 2**32 - 1)
            view_idx = 0
            seed_sampler = random.randint(0, 2**32 - 1)
            process_multiview(person_rel, cloth_rel, output_subdir, basename, view_idx, seed_qwen, seed_sampler)
            print(f"    View {view_idx + 1}/8 Done")
            success += 1
        except Exception as e:
            fail += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"COMPLETED: OK {success} FAIL {fail}")


if __name__ == "__main__":
    main()
