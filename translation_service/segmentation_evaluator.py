#!/usr/bin/env python3
"""
Segmentation Evaluator
Scans output/optimized_results/symbols/*/results.json and produces per-image and overall metrics
Exports: CSV, Markdown, LaTeX tables for academic paper
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, List, Any
from statistics import mean, median
import pandas as pd

OUTPUT_ROOT = Path("output/optimized_results/symbols")
REPORTS_DIR = Path("output/optimized_results")

def _load_results_file(results_path: Path) -> Dict[str, Any]:
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _bin_confidence(c: float) -> str:
    if c is None:
        return "unknown"
    if c > 0.7:
        return "high"
    if c > 0.4:
        return "medium"
    return "low"

def compute_image_metrics(results_json: Dict[str, Any], symbols_dir: Path) -> Dict[str, Any]:
    classes = results_json.get('classifications', []) or []
    confidences = [float(cls.get('confidence', 0) or 0) for cls in classes if 'error' not in cls]

    low = sum(1 for c in confidences if c <= 0.4)
    med = sum(1 for c in confidences if 0.4 < c <= 0.7)
    high = sum(1 for c in confidences if c > 0.7)

    crops = list((symbols_dir.glob('*.png')))

    metrics = {
        'image_name': Path(results_json.get('image_path', '')).name or symbols_dir.parent.name,
        'symbols_found_field': int(results_json.get('symbols_found', len(classes))),
        'num_classifications': len(classes),
        'num_crops': len(crops),
        'processing_time_s': float(results_json.get('processing_time', 0.0)),
        'conf_mean': float(mean(confidences)) if confidences else 0.0,
        'conf_median': float(median(confidences)) if confidences else 0.0,
        'conf_min': float(min(confidences)) if confidences else 0.0,
        'conf_max': float(max(confidences)) if confidences else 0.0,
        'conf_low': low,
        'conf_medium': med,
        'conf_high': high,
    }
    return metrics

def load_all_segmentation_results(root: Path) -> List[Dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(f"Segmentation root not found: {root}")

    per_image = []
    for image_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not image_dir.is_dir():
            continue
        results_path = image_dir / 'results.json'
        symbols_dir = image_dir / 'symbols'
        if not results_path.exists() or not symbols_dir.exists():
            continue
        try:
            results_json = _load_results_file(results_path)
            metrics = compute_image_metrics(results_json, symbols_dir)
            per_image.append(metrics)
        except Exception as e:
            # Skip corrupted entries but continue
            print(f"Warning: failed processing {image_dir.name}: {e}")
            continue
    return per_image

def export_csv(per_image: List[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_image)
    csv_path = out_dir / 'segmentation_overview.csv'
    df.to_csv(csv_path, index=False)
    return csv_path

def export_markdown(per_image: List[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("## Segmentation Overview\n")
    lines.append("| Image | Symbols (field) | Classifications | Crops | Time (s) | Mean | Median | Min | Max | Low | Med | High |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for m in per_image:
        lines.append(
            f"| {m['image_name']} | {m['symbols_found_field']} | {m['num_classifications']} | {m['num_crops']} | "
            f"{m['processing_time_s']:.2f} | {m['conf_mean']:.2f} | {m['conf_median']:.2f} | {m['conf_min']:.2f} | {m['conf_max']:.2f} | "
            f"{m['conf_low']} | {m['conf_medium']} | {m['conf_high']} |\n"
        )

    md_path = out_dir / 'segmentation_overview.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return md_path

def export_latex(per_image: List[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Segmentation Results Per Image}\n"
        "\\begin{tabular}{lrrrrrrrrrrr}\n"
        "\\toprule\n"
        "Image & Sym(field) & Cls & Crops & Time(s) & Mean & Median & Min & Max & Low & Med & High \\\\\n"
        "\\midrule\n"
    )
    rows = []
    for m in per_image:
        rows.append(
            f"{m['image_name']} & {m['symbols_found_field']} & {m['num_classifications']} & {m['num_crops']} & "
            f"{m['processing_time_s']:.2f} & {m['conf_mean']:.2f} & {m['conf_median']:.2f} & {m['conf_min']:.2f} & {m['conf_max']:.2f} & "
            f"{m['conf_low']} & {m['conf_medium']} & {m['conf_high']} \\\n"
        )
    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    tex_content = header + ''.join(rows) + footer
    tex_path = out_dir / 'segmentation_tables.tex'
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    return tex_path

def main():
    per_image = load_all_segmentation_results(OUTPUT_ROOT)
    if not per_image:
        print(f"No segmentation results found under {OUTPUT_ROOT}")
        return 1
    csv_path = export_csv(per_image, REPORTS_DIR)
    md_path = export_markdown(per_image, REPORTS_DIR)
    tex_path = export_latex(per_image, REPORTS_DIR)
    print("Segmentation evaluation generated:")
    print(f"- CSV: {csv_path}")
    print(f"- Markdown: {md_path}")
    print(f"- LaTeX: {tex_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
