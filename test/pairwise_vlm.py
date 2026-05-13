#!/usr/bin/env python3
"""
Pairwise VLM comparison: idmvton vs ours_stage2 only.
Random candidate assignment, no method name leakage.
"""

import os, sys, json, csv, argparse, random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import numpy as np

from PIL import Image
import torch

VLM_PATH = "/hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI/models/LLM/Qwen-VL/Qwen3-VL-4B-Instruct"
DATASET_CSV = "/hpc2hdd/home/dsaa2012_017/dl_test/data/dataset.csv"
OUTPUT_DIR = Path("/hpc2hdd/home/dsaa2012_017/dl_test/eval_results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_A = "idmvton"
METHOD_B = "ours_stage2"

def load_image(path: str) -> Image.Image:
    """Load and normalize image format for VLM."""
    img = Image.open(path).convert("RGB")
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return Image.open(buf).convert('RGB')

class VLMPairwise:
    def __init__(self):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        print("Loading VLM...", flush=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            VLM_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(VLM_PATH, trust_remote_code=True)
        self.rng = random.Random(42)
        print("VLM loaded.", flush=True)

    @torch.no_grad()
    def compare(self, person_img: Image.Image, garment_img: Image.Image,
                output_a: Image.Image, output_b: Image.Image,
                method_a: str, method_b: str) -> Dict:
        """Randomly assign to candidate_1/2 and query VLM."""

        # Randomize order
        if self.rng.random() < 0.5:
            cand1_img, cand2_img = output_a, output_b
            cand1_method, cand2_method = method_a, method_b
            swap = False
        else:
            cand1_img, cand2_img = output_b, output_a
            cand1_method, cand2_method = method_b, method_a
            swap = True

        prompt = (
            "You are comparing two virtual try-on results. The first image is the original person, "
            "the second image is the target garment. The third and fourth images are two candidate try-on results.\n"
            "Compare Candidate 1 (third image) and Candidate 2 (fourth image) on the following dimensions:\n"
            "1. garment_fidelity: Which better matches the target garment? (category, color, pattern, logo, structure)\n"
            "2. character_fidelity: Which better preserves the original person? (face, hair, pose, skin tone)\n"
            "3. harmony_realism: Which looks more natural and artifact-free? (fit, lighting, boundaries)\n"
            "4. overall_preference: Which is better overall for virtual try-on?\n"
            'Return JSON only: {"garment_fidelity": {"winner": "candidate_1"|"candidate_2"|"tie", "reason": "brief"}, '
            '"character_fidelity": {"winner": "...", "reason": "..."}, '
            '"harmony_realism": {"winner": "...", "reason": "..."}, '
            '"overall_preference": {"winner": "...", "reason": "..."}}'
        )

        messages = [{"role": "user", "content": [
            {"type": "image", "image": person_img},
            {"type": "image", "image": garment_img},
            {"type": "image", "image": cand1_img},
            {"type": "image", "image": cand2_img},
            {"type": "text", "text": prompt},
        ]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[person_img, garment_img, cand1_img, cand2_img], return_tensors="pt")
        target_device = self.model.device
        target_dtype = self.model.dtype
        inputs = {
            k: v.to(device=target_device, dtype=target_dtype if isinstance(v, torch.Tensor) and v.is_floating_point() else None)
            if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        generated_ids = self.model.generate(**inputs, max_new_tokens=512, temperature=0.0, top_p=1.0, do_sample=False)
        generated_ids = [oid[len(iid):] for oid, iid in zip(generated_ids, inputs['input_ids'])]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        result = {"swap": swap, "candidate_1_method": cand1_method, "candidate_2_method": cand2_method}

        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(response[json_start:json_end])
                result["vlm_raw"] = parsed
                # Map candidate_1/candidate_2 to actual method names
                for dim in ['garment_fidelity', 'character_fidelity', 'harmony_realism', 'overall_preference']:
                    if dim in parsed:
                        winner = parsed[dim].get('winner', '?')
                        reason = parsed[dim].get('reason', '')
                        if winner == 'candidate_1':
                            actual_winner = cand1_method
                        elif winner == 'candidate_2':
                            actual_winner = cand2_method
                        else:
                            actual_winner = 'tie'
                        result[f"{dim}_winner"] = actual_winner
                        result[f"{dim}_reason"] = reason[:300]
            else:
                result["error"] = "json_parse_failed"
                result["raw_response"] = response[:300]
        except (json.JSONDecodeError, ValueError) as e:
            result["error"] = f"json_parse_error: {e}"
            result["raw_response"] = response[:300]

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk', type=int, default=-1)
    parser.add_argument('--total-chunks', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--method-a', type=str, default='idmvton')
    parser.add_argument('--method-b', type=str, default='ours_stage2')
    args = parser.parse_args()

    method_a = args.method_a
    method_b = args.method_b

    # Load CSV, filter to target methods only
    all_rows = []
    with open(DATASET_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['model'] in (method_a, method_b):
                all_rows.append(row)

    print(f"Filtered to {method_a} + {method_b}: {len(all_rows)} rows")

    # Group by sample_id into pairs
    by_sample = defaultdict(dict)
    for row in all_rows:
        by_sample[row['sample_id']][row['model']] = row

    # Keep only samples that have BOTH methods
    paired_samples = []
    for sid, methods in sorted(by_sample.items()):
        if method_a in methods and method_b in methods:
            paired_samples.append((sid, methods[method_a], methods[method_b]))
    
    print(f"Samples with both methods: {len(paired_samples)}")

    # Apply chunking or start/end
    if args.chunk >= 0:
        chunk_size = (len(paired_samples) + args.total_chunks - 1) // args.total_chunks
        cs = args.chunk * chunk_size
        ce = min(cs + chunk_size, len(paired_samples))
        paired_samples = paired_samples[cs:ce]
        chunk_tag = f"chunk{args.chunk}"
        print(f"Chunk {args.chunk}/{args.total_chunks}: samples {cs}-{ce} ({len(paired_samples)})")
    else:
        if args.end < 0:
            args.end = len(paired_samples)
        paired_samples = paired_samples[args.start:args.end]
        chunk_tag = f"{args.start}_{args.end}"
        print(f"Samples {args.start}-{args.end}: {len(paired_samples)}")

    if len(paired_samples) == 0:
        print("No samples to process!")
        return

    # Init VLM
    vlm = VLMPairwise()

    # Storage
    results = []
    person_img_cache = {}
    garment_img_cache = {}

    from tqdm import tqdm
    for sid, row_a, row_b in tqdm(paired_samples, desc="Pairwise"):
        person_path = row_a['person_image_path']
        garment_path = row_a['garment_image_path']
        
        # Cache person/garment images
        if person_path not in person_img_cache:
            person_img_cache[person_path] = load_image(person_path)
        if garment_path not in garment_img_cache:
            garment_img_cache[garment_path] = load_image(garment_path)
        
        person_img = person_img_cache[person_path]
        garment_img = garment_img_cache[garment_path]

        try:
            img_a = load_image(row_a['model_output_path'])
            img_b = load_image(row_b['model_output_path'])
            
            result = vlm.compare(person_img, garment_img, img_a, img_b, method_a, method_b)
            result['sample_id'] = sid
            result['person_name'] = row_a['person_name']
            result['cloth_name'] = row_a['cloth_name']
            results.append(result)
        except Exception as e:
            results.append({
                'sample_id': sid,
                'person_name': row_a['person_name'],
                'cloth_name': row_a['cloth_name'],
                'error': str(e)[:200],
            })

    # Compute win rates
    wins_a = {'garment_fidelity': 0, 'character_fidelity': 0, 'harmony_realism': 0, 'overall_preference': 0}
    wins_b = {'garment_fidelity': 0, 'character_fidelity': 0, 'harmony_realism': 0, 'overall_preference': 0}
    ties = {'garment_fidelity': 0, 'character_fidelity': 0, 'harmony_realism': 0, 'overall_preference': 0}
    valid = 0

    for r in results:
        if 'error' in r:
            continue
        valid += 1
        for dim in ['garment_fidelity', 'character_fidelity', 'harmony_realism', 'overall_preference']:
            winner = r.get(f'{dim}_winner', '?')
            if winner == method_a:
                wins_a[dim] += 1
            elif winner == method_b:
                wins_b[dim] += 1
            else:
                ties[dim] += 1

    print(f"\n=== Pairwise Results: {method_a} vs {method_b} ===")
    print(f"Valid comparisons: {valid}/{len(results)}")
    print(f"{'Dimension':<25} {method_a:>8} {method_b:>12} {'tie':>6} {'win_rate('+method_b+')':>14}")
    print("-" * 68)
    for dim in ['garment_fidelity', 'character_fidelity', 'harmony_realism', 'overall_preference']:
        wr = wins_b[dim] / max(valid, 1)
        print(f"{dim:<25} {wins_a[dim]:>8} {wins_b[dim]:>12} {ties[dim]:>6} {wr:>13.1%}")

    # Save results
    csv_path = OUTPUT_DIR / f"pairwise_{method_a}_vs_{method_b}_{chunk_tag}.csv"
    fieldnames = ['sample_id', 'person_name', 'cloth_name',
                  'swap', 'candidate_1_method', 'candidate_2_method',
                  'garment_fidelity_winner', 'garment_fidelity_reason',
                  'character_fidelity_winner', 'character_fidelity_reason',
                  'harmony_realism_winner', 'harmony_realism_reason',
                  'overall_preference_winner', 'overall_preference_reason',
                  'vlm_raw', 'error']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    # Save summary
    summary = {
        'method_a': method_a, 'method_b': method_b,
        'valid': valid, 'total': len(results),
        'win_rate': {dim: round(wins_b[dim]/max(valid,1), 4) for dim in wins_b},
        'detail': {dim: {method_a: wins_a[dim], method_b: wins_b[dim], 'tie': ties[dim]} for dim in wins_b},
    }
    with open(OUTPUT_DIR / f"pairwise_summary_{method_a}_vs_{method_b}_{chunk_tag}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {csv_path}")

if __name__ == '__main__':
    main()
