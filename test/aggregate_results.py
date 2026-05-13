#!/usr/bin/env python3
"""
Aggregate evaluation results into summary tables.
"""
import sys, os, json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np

METRICS_DIR = Path("/hpc2hdd/home/dsaa2012_017/dl_test/eval_results/metrics")

def load_csvs(pattern):
    """Load all CSV files matching pattern and concatenate."""
    rows = []
    for f in sorted(METRICS_DIR.glob(pattern)):
        with open(f, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    return rows

def float_or_none(v):
    if v is None or v == '' or v == 'None':
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def compute_summary(rows, metric_keys, group_key='model'):
    """Compute mean, std, and count for each metric per group."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        g = row[group_key]
        for mk in metric_keys:
            val = float_or_none(row.get(mk))
            if val is not None:
                groups[g][mk].append(val)
    
    summary = []
    for g, metrics in sorted(groups.items()):
        entry = {'Method': g}
        for mk in metric_keys:
            vals = metrics.get(mk, [])
            if vals:
                entry[f'{mk}_mean'] = round(np.mean(vals), 4)
                entry[f'{mk}_std'] = round(np.std(vals), 4)
                entry[f'{mk}_n'] = len(vals)
            else:
                entry[f'{mk}_mean'] = None
                entry[f'{mk}_std'] = None
                entry[f'{mk}_n'] = 0
        summary.append(entry)
    return summary

def compute_bootstrap_ci(values, n_bootstrap=1000, alpha=0.05):
    """Compute 95% bootstrap CI for mean."""
    if len(values) < 5:
        return None, None
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    means = np.sort(means)
    lower = means[int(alpha/2 * n_bootstrap)]
    upper = means[int((1-alpha/2) * n_bootstrap)]
    return round(lower, 4), round(upper, 4)

def main():
    print("=== Aggregating Evaluation Results ===\n")
    
    # 1. Preservation metrics
    pres_rows = load_csvs("preservation_metrics_*.csv")
    print(f"Preservation metrics rows: {len(pres_rows)}")
    
    pres_metrics = [
        'Masked_SSIM_person_noncloth',
        'Masked_LPIPS_person_noncloth',
        'Masked_PSNR_person_noncloth',
        'Masked_SSIM_full_noncloth',
        'Masked_LPIPS_full_noncloth',
        'Masked_PSNR_full_noncloth',
    ]
    
    pres_summary = compute_summary(pres_rows, pres_metrics)
    
    # 2. Garment fidelity metrics
    garm_rows = load_csvs("garment_fidelity_metrics_*.csv")
    print(f"Garment metrics rows: {len(garm_rows)}")
    
    garm_metrics = ['CLIP_Garment_Similarity']
    garm_summary = compute_summary(garm_rows, garm_metrics)
    
    # 3. VLM garment scores
    vlm_g_rows = load_csvs("vlm_garment_scores_*.csv")
    print(f"VLM garment score rows: {len(vlm_g_rows)}")
    
    vlm_g_metrics = [
        'VLM_category_consistency',
        'VLM_color_consistency',
        'VLM_pattern_logo_consistency',
        'VLM_shape_structure_consistency',
        'VLM_overall_garment_fidelity',
    ]
    vlm_g_summary = compute_summary(vlm_g_rows, vlm_g_metrics)
    
    # 4. VLM preference scores
    vlm_p_rows = load_csvs("vlm_preference_scores_*.csv")
    print(f"VLM preference score rows: {len(vlm_p_rows)}")
    
    vlm_p_metrics = [
        'VLM_garment_fidelity',
        'VLM_character_fidelity',
        'VLM_harmony_realism',
        'VLM_overall_quality',
    ]
    vlm_p_summary = compute_summary(vlm_p_rows, vlm_p_metrics)
    
    # Combine all summaries
    methods = sorted(set(
        [r['Method'] for r in pres_summary] + 
        [r['Method'] for r in garm_summary] + 
        [r['Method'] for r in vlm_g_summary] + 
        [r['Method'] for r in vlm_p_summary]
    ))
    
    # Build consolidated summary
    consolidated = {}
    for m in methods:
        consolidated[m] = {'Method': m}
    
    for sr in pres_summary:
        m = sr['Method']
        for k, v in sr.items():
            if k != 'Method':
                consolidated[m][k] = v
    
    for sr in garm_summary:
        m = sr['Method']
        for k, v in sr.items():
            if k != 'Method':
                consolidated[m][k] = v
    
    for sr in vlm_g_summary:
        m = sr['Method']
        for k, v in sr.items():
            if k != 'Method':
                consolidated[m][k] = v
    
    for sr in vlm_p_summary:
        m = sr['Method']
        for k, v in sr.items():
            if k != 'Method':
                consolidated[m][k] = v
    
    # Print main table
    print("\n=== Main Result Table ===")
    print(f"{'Method':<15} {'SSIM↑':>8} {'LPIPS↓':>8} {'PSNR↑':>8} {'CLIP-G↑':>8} {'VLM-GF↑':>8} {'VLM-CF↑':>8} {'VLM-HR↑':>8} {'VLM-OQ↑':>8}")
    print("-" * 100)
    
    for m in sorted(consolidated.keys()):
        d = consolidated[m]
        ssim = f"{d.get('Masked_SSIM_person_noncloth_mean','?'):.4f}" if d.get('Masked_SSIM_person_noncloth_mean') else "N/A"
        lpips = f"{d.get('Masked_LPIPS_person_noncloth_mean','?'):.4f}" if d.get('Masked_LPIPS_person_noncloth_mean') else "N/A"
        psnr = f"{d.get('Masked_PSNR_person_noncloth_mean','?'):.2f}" if d.get('Masked_PSNR_person_noncloth_mean') else "N/A"
        clipg = f"{d.get('CLIP_Garment_Similarity_mean','?'):.4f}" if d.get('CLIP_Garment_Similarity_mean') else "N/A"
        vlmgf = f"{d.get('VLM_overall_garment_fidelity_mean','?'):.2f}" if d.get('VLM_overall_garment_fidelity_mean') else "N/A"
        vlmcf = f"{d.get('VLM_character_fidelity_mean','?'):.2f}" if d.get('VLM_character_fidelity_mean') else "N/A"
        vlmhr = f"{d.get('VLM_harmony_realism_mean','?'):.2f}" if d.get('VLM_harmony_realism_mean') else "N/A"
        vlmoq = f"{d.get('VLM_overall_quality_mean','?'):.2f}" if d.get('VLM_overall_quality_mean') else "N/A"
        
        print(f"{m:<15} {ssim:>8} {lpips:>8} {psnr:>8} {clipg:>8} {vlmgf:>8} {vlmcf:>8} {vlmhr:>8} {vlmoq:>8}")
    
    # Save detailed summary JSON
    summary_json = {
        'methods': sorted(consolidated.keys()),
        'results': {m: {k: v for k, v in d.items()} for m, d in consolidated.items()},
        'num_samples': len(set(r['sample_id'] for r in pres_rows)) if pres_rows else 0,
    }
    
    with open(METRICS_DIR / "summary.json", 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"\nSummary saved to {METRICS_DIR / 'summary.json'}")
    
    # Invalid cases
    inv_rows = load_csvs("invalid_cases_*.csv")
    print(f"\nInvalid cases: {len(inv_rows)}")
    if inv_rows:
        with open(METRICS_DIR / "invalid_cases_all.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=inv_rows[0].keys())
            writer.writeheader()
            writer.writerows(inv_rows)

if __name__ == '__main__':
    main()
