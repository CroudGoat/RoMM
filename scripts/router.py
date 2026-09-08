#!/usr/bin/env python3
"""Routing of Model Merging (RoMM) Router & Parameter Calibrator.

High-performance, ultra-low-memory implementation using safetensors get_slice()
and BLAS-accelerated vector operations with dynamic output naming and YAML flags.

Dependencies:
    pip install torch safetensors huggingface_hub pyyaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

DEFAULT_LAYER_PATTERN = r"(?:^|\.)layers\.(\d+)(?:\.|$)"
INDEX_FILENAME = "model.safetensors.index.json"


def sanitize_identifier(name: str) -> str:
    """Sanitize model ID or path to a filesystem-safe string."""
    clean = name.strip("/").replace("\\", "/")
    # Extract the last 1 or 2 parts for brevity and uniqueness (e.g. 'org__model' or 'model')
    parts = [p for p in clean.split("/") if p]
    target = "__".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return re.sub(r"[^\w\-.]", "_", target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RoMM: High-performance Layer-wise Model Merging Router."
    )
    parser.add_argument("--base", required=True, help="Base model repo ID or local path")
    parser.add_argument("--model-a", required=True, help="Model A repo ID or local path")
    parser.add_argument("--model-b", required=True, help="Model B repo ID or local path")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base directory for outputs (default: results)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Explicit CSV output path (overrides dynamic naming)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Explicit JSON output path (overrides dynamic naming)",
    )
    parser.add_argument(
        "--output-yaml",
        default=None,
        help="Explicit YAML output path (overrides dynamic naming)",
    )
    # YAML generation flag (--yaml / --no-yaml)
    parser.add_argument(
        "--yaml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Mergekit YAML config (default: --yaml, use --no-yaml to disable)",
    )
    parser.add_argument("--cache-dir", default=None, help="HF cache directory")
    parser.add_argument("--offline", action="store_true", help="Local cache only")
    parser.add_argument(
        "--chunk-elements",
        type=int,
        default=2_000_000,
        help="Chunk size for tensor streaming",
    )
    parser.add_argument(
        "--layer-pattern",
        default=DEFAULT_LAYER_PATTERN,
        help="Regex for layer index",
    )
    parser.add_argument(
        "--ortho-min",
        type=float,
        default=-0.002,
        help="Orthogonal range min",
    )
    parser.add_argument(
        "--ortho-max",
        type=float,
        default=0.002,
        help="Orthogonal range max",
    )
    parser.add_argument(
        "--base-density",
        type=float,
        default=0.20,
        help="Base density (D_base)",
    )
    parser.add_argument(
        "--alpha-a",
        type=float,
        default=1.0,
        help="Model A user priority",
    )
    parser.add_argument(
        "--alpha-b",
        type=float,
        default=1.0,
        help="Model B user priority",
    )
    return parser.parse_args()


def resolve_model(source: str, cache_dir: Optional[str], offline: bool) -> Path:
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    try:
        resolved = snapshot_download(
            repo_id=source,
            local_files_only=offline,
            allow_patterns=["*.safetensors", "*.json"],
            cache_dir=str(Path(cache_dir).expanduser()) if cache_dir else None,
        )
        return Path(resolved).resolve()
    except Exception as exc:
        raise RuntimeError(f"Failed to resolve '{source}': {exc}") from exc


def build_key_to_file_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / INDEX_FILENAME
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        return {k: model_dir / v for k, v in weight_map.items()}

    result = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                result[key] = sf
    if not result:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    return result


def extract_common_layers(
    maps: list[Mapping[str, Path]], pattern: str
) -> dict[int, list[str]]:
    layer_re = re.compile(pattern)
    layer_keys_per_model = []
    for m in maps:
        lk = defaultdict(list)
        for k in m:
            match = layer_re.search(k)
            if match:
                lk[int(match.group(1))].append(k)
        layer_keys_per_model.append(lk)

    common_layers = sorted(
        set(layer_keys_per_model[0])
        & set(layer_keys_per_model[1])
        & set(layer_keys_per_model[2])
    )
    if not common_layers:
        raise RuntimeError(f"No common layers found matching pattern: {pattern}")

    merged = {}
    for n in common_layers:
        keys = sorted(
            set(layer_keys_per_model[0][n])
            & set(layer_keys_per_model[1][n])
            & set(layer_keys_per_model[2][n])
        )
        if keys:
            merged[n] = keys
    return merged


class SliceReader:
    """Zero-copy streaming slice reader for memory efficiency."""

    def __init__(self, key_map: Mapping[str, Path]):
        self.key_map = key_map
        self._handles: dict[Path, Any] = {}

    def get_slice(self, key: str):
        path = self.key_map[key]
        if path not in self._handles:
            self._handles[path] = safe_open(str(path), framework="pt", device="cpu")
        return self._handles[path].get_slice(key)

    def close(self):
        for h in self._handles.values():
            close_fn = getattr(h, "__exit__", None)
            if close_fn:
                close_fn(None, None, None)
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def stream_tensor_stats(
    base_slice, a_slice, b_slice, chunk_elements: int
) -> tuple[float, float, float, int]:
    shape = base_slice.get_shape()
    if shape != a_slice.get_shape() or shape != b_slice.get_shape():
        raise ValueError(f"Shape mismatch: {shape}")

    dim0 = shape[0]
    row_elements = math.prod(shape[1:]) if len(shape) > 1 else 1
    total_elements = dim0 * row_elements
    rows_per_chunk = max(1, chunk_elements // row_elements)

    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for r in range(0, dim0, rows_per_chunk):
        r_end = min(r + rows_per_chunk, dim0)

        base_chunk = base_slice[r:r_end].float().reshape(-1)
        da = (a_slice[r:r_end].float().reshape(-1) - base_chunk).double()
        db = (b_slice[r:r_end].float().reshape(-1) - base_chunk).double()

        dot += torch.dot(da, db).item()
        norm_a_sq += torch.dot(da, da).item()
        norm_b_sq += torch.dot(db, db).item()

        del base_chunk, da, db

    return dot, norm_a_sq, norm_b_sq, total_elements


def sanitize(val: Optional[float]) -> Optional[float]:
    return None if (val is None or math.isnan(val) or math.isinf(val)) else round(val, 6)


def main() -> int:
    args = parse_args()

    # Dynamic naming resolution
    tag_base = sanitize_identifier(args.base)
    tag_a = sanitize_identifier(args.model_a)
    tag_b = sanitize_identifier(args.model_b)
    pair_slug = f"{tag_base}__{tag_a}_x_{tag_b}"

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else out_dir / f"{pair_slug}_routing.csv"
    )
    json_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else out_dir / f"{pair_slug}_routing.json"
    )
    yaml_path = (
        Path(args.output_yaml).expanduser().resolve()
        if args.output_yaml
        else out_dir / f"{pair_slug}_mergekit.yaml"
    )

    print("[1/4] Resolving models and building index...")
    base_dir = resolve_model(args.base, args.cache_dir, args.offline)
    a_dir = resolve_model(args.model_a, args.cache_dir, args.offline)
    b_dir = resolve_model(args.model_b, args.cache_dir, args.offline)

    base_map = build_key_to_file_map(base_dir)
    a_map = build_key_to_file_map(a_dir)
    b_map = build_key_to_file_map(b_dir)

    layers = extract_common_layers([base_map, a_map, b_map], args.layer_pattern)
    print(f"  Common layers: {len(layers)} (L{min(layers)}..L{max(layers)})")

    print("[2/4] Streaming geometric feature extraction (Low-RAM & BLAS)...")
    raw_stats = []
    all_norms = []

    with SliceReader(base_map) as br, SliceReader(a_map) as ar, SliceReader(b_map) as lr:
        for layer_id, keys in layers.items():
            dot = norm_a_sq = norm_b_sq = 0.0
            layer_elements = 0

            for key in keys:
                d, na, nb, count = stream_tensor_stats(
                    br.get_slice(key), ar.get_slice(key), lr.get_slice(key), args.chunk_elements
                )
                dot += d
                norm_a_sq += na
                norm_b_sq += nb
                layer_elements += count

            denom = math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq)
            cosine = max(-1.0, min(1.0, dot / denom)) if denom > 0.0 else 0.0
            norm_a = math.sqrt(max(norm_a_sq, 0.0))
            norm_b = math.sqrt(max(norm_b_sq, 0.0))

            all_norms.extend([norm_a, norm_b])
            raw_stats.append((layer_id, cosine, norm_a, norm_b, layer_elements, len(keys)))

    mean_norm = sum(all_norms) / len(all_norms) if all_norms else 1.0

    print("[3/4] Routing and parameter calibration...")
    results = []
    for layer_id, score, norm_a, norm_b, elements, param_count in raw_stats:
        if score > args.ortho_max:
            cat, method = "High-Similarity", "slerp"
        elif score < args.ortho_min:
            cat, method = "Conflicting", "dare_ties"
        else:
            cat, method = "Orthogonal", "dare_ties"

        slerp_t = norm_a / (norm_a + norm_b) if (norm_a + norm_b) > 0 else 0.5

        w_denom = (args.alpha_a * norm_b) + (args.alpha_b * norm_a)
        w_a = (args.alpha_a * norm_b) / w_denom if w_denom > 0 else 0.5
        w_b = (args.alpha_b * norm_a) / w_denom if w_denom > 0 else 0.5

        ortho_factor = 1.0 + abs(score)
        d_a = max(0.05, min(args.base_density * ortho_factor * (mean_norm / norm_a if norm_a > 0 else 1.0), 0.50))
        d_b = max(0.05, min(args.base_density * ortho_factor * (mean_norm / norm_b if norm_b > 0 else 1.0), 0.50))

        results.append({
            "layer": layer_id,
            "cosine_similarity": score,
            "category": cat,
            "recommended_method": method,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "norm_ratio_a_over_b": (norm_a / norm_b) if norm_b > 0 else None,
            "slerp_t": slerp_t,
            "dare_weight_a": w_a,
            "dare_weight_b": w_b,
            "dare_density_a": d_a,
            "dare_density_b": d_b,
            "parameter_tensors": param_count,
            "elements": elements,
        })

    # Display Table
    print("\n" + "=" * 115)
    print(f" RoMM Routing Summary (Ortho: [{args.ortho_min}, {args.ortho_max}], Mean ||v||: {mean_norm:.4f})")
    print("=" * 115)
    print(f"{'Layer':<6} | {'Cosine':<12} | {'Category':<16} | {'Method':<10} | {'||dA||':<8} | {'||dB||':<8} | Calibrated Parameters")
    print("-" * 115)
    for r in results:
        param_str = (
            f"t={r['slerp_t']:.4f}"
            if r["recommended_method"] == "slerp"
            else f"w=[A:{r['dare_weight_a']:.3f}, B:{r['dare_weight_b']:.3f}] d=[A:{r['dare_density_a']:.3f}, B:{r['dare_density_b']:.3f}]"
        )
        print(f"L{r['layer']:<5} | {r['cosine_similarity']:+10.6f} | {r['category']:<16} | {r['recommended_method'].upper():<10} | {r['norm_a']:<8.4f} | {r['norm_b']:<8.4f} | {param_str}")
    print("=" * 115 + "\n")

    # Write CSV & JSON
    print("[4/4] Writing output files...")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV : {csv_path}")

    json_data = {
        "framework": "RoMM",
        "models": {
            "base": args.base,
            "model_a": args.model_a,
            "model_b": args.model_b,
            "resolved_base": str(base_dir),
            "resolved_model_a": str(a_dir),
            "resolved_model_b": str(b_dir),
        },
        "config": {
            "ortho_range": [args.ortho_min, args.ortho_max],
            "base_density": args.base_density,
            "alpha_a": args.alpha_a,
            "alpha_b": args.alpha_b,
            "mean_norm": sanitize(mean_norm),
        },
        "layers": [
            {k: (sanitize(v) if isinstance(v, float) else v) for k, v in r.items()}
            for r in results
        ],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path}")

    # Conditional YAML Generation
    if args.yaml:
        slices = []
        for r in results:
            idx = r["layer"]
            if r["recommended_method"] == "slerp":
                slices.append({
                    "sources": [
                        {"model": args.model_a, "layer_range": [idx, idx + 1]},
                        {"model": args.model_b, "layer_range": [idx, idx + 1]},
                    ],
                    "merge_method": "slerp",
                    "parameters": {"t": round(r["slerp_t"], 4)},
                })
            else:
                slices.append({
                    "sources": [
                        {
                            "model": args.model_a,
                            "layer_range": [idx, idx + 1],
                            "parameters": {
                                "weight": round(r["dare_weight_a"], 4),
                                "density": round(r["dare_density_a"], 4),
                            },
                        },
                        {
                            "model": args.model_b,
                            "layer_range": [idx, idx + 1],
                            "parameters": {
                                "weight": round(r["dare_weight_b"], 4),
                                "density": round(r["dare_density_b"], 4),
                            },
                        },
                    ],
                    "merge_method": "dare_ties",
                })

        try:
            import yaml
            with yaml_path.open("w", encoding="utf-8") as f:
                yaml.dump({"base_model": args.base, "dtype": "bfloat16", "slices": slices}, f, sort_keys=False)
        except ImportError:
            with yaml_path.open("w", encoding="utf-8") as f:
                f.write(f"base_model: {args.base}\ndtype: bfloat16\nslices:\n")
                for s in slices:
                    f.write(f"  - merge_method: {s['merge_method']}\n")
                    if "parameters" in s:
                        f.write(f"    parameters:\n      t: {s['parameters']['t']}\n")
                    f.write("    sources:\n")
                    for src in s["sources"]:
                        f.write(f"      - model: {src['model']}\n        layer_range: {src['layer_range']}\n")
                        if "parameters" in src:
                            f.write(f"        parameters:\n          weight: {src['parameters']['weight']}\n          density: {src['parameters']['density']}\n")
        print(f"  YAML: {yaml_path}")
    else:
        print("  YAML: Skipped (--no-yaml specified)")

    print("Execution complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)