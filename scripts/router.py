#!/usr/bin/env python3
"""Routing of Model Merging (RoMM) Router & Parameter Calibrator.

High-performance, low-memory implementation using safetensors streaming,
Stable Rank estimation via Power Iteration, and dimension-aware statistical routing.

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
    parts = [p for p in clean.split("/") if p]
    target = "__".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return re.sub(r"[^\w\-.]", "_", target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RoMM: Dimension-Aware Statistical Model Merging Router."
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
    # Statistical & Effective Dimension options (Calibrated for Stable Rank)
    parser.add_argument(
        "--use-stable-rank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Effective Dimension via Stable Rank instead of physical elements (default: True)",
    )
    parser.add_argument(
        "--power-iters",
        type=int,
        default=5,
        help="Number of power iterations for spectral norm estimation (default: 5)",
    )
    parser.add_argument(
        "--z-slerp",
        type=float,
        default=2.5,
        help="Z-score threshold for High-Similarity (SLERP) routing (default: 2.5 sigma)",
    )
    parser.add_argument(
        "--z-conflict",
        type=float,
        default=2.0,
        help="Z-score threshold for Conflicting (Strong-Sparse) routing (default: 2.0 sigma, z < -2.0)",
    )
    parser.add_argument(
        "--base-density",
        type=float,
        default=0.20,
        help="Base density (D_base) for DARE-TIES (default: 0.20)",
    )
    parser.add_argument(
        "--alpha-a",
        type=float,
        default=1.0,
        help="Model A user priority multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--alpha-b",
        type=float,
        default=1.0,
        help="Model B user priority multiplier (default: 1.0)",
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


def estimate_spectral_norm_power_iter(mat: torch.Tensor, iters: int = 5) -> float:
    """Ultra-fast spectral norm (sigma_max) estimation using power iteration."""
    if mat.ndim == 1:
        return torch.norm(mat, p=2).item()

    m2d = mat.flatten(1).float()
    out_dim, in_dim = m2d.shape
    if out_dim == 0 or in_dim == 0:
        return 0.0

    generator = torch.Generator(device="cpu").manual_seed(42)
    v = torch.randn(in_dim, 1, generator=generator, dtype=m2d.dtype)
    v = v / (torch.norm(v) + 1e-12)

    for _ in range(iters):
        u = torch.matmul(m2d, v)
        u_norm = torch.norm(u)
        if u_norm < 1e-12:
            return 0.0
        u = u / u_norm

        v = torch.matmul(m2d.T, u)
        v_norm = torch.norm(v)
        if v_norm < 1e-12:
            return 0.0
        v = v / v_norm

    sigma_max = torch.norm(torch.matmul(m2d, v)).item()
    return sigma_max


def compute_tensor_stable_rank(delta: torch.Tensor, iters: int = 5) -> float:
    """Calculate Stable Rank = ||W||_F^2 / ||W||_2^2."""
    fro_sq = torch.sum(delta.float() ** 2).item()
    if fro_sq < 1e-12:
        return 1.0

    sigma_max = estimate_spectral_norm_power_iter(delta, iters=iters)
    sigma_max_sq = sigma_max ** 2

    if sigma_max_sq < 1e-12:
        return 1.0

    srank = fro_sq / sigma_max_sq
    return max(1.0, srank)


def stream_tensor_stats(
    base_slice, a_slice, b_slice, chunk_elements: int, power_iters: int = 5
) -> tuple[float, float, float, int, float, float]:
    """Extract dot, norms, element count, and stable ranks via power iteration."""
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

    t_base = base_slice[:].float()
    t_da = a_slice[:].float() - t_base
    t_db = b_slice[:].float() - t_base
    del t_base

    srank_a = compute_tensor_stable_rank(t_da, iters=power_iters)
    srank_b = compute_tensor_stable_rank(t_db, iters=power_iters)
    del t_da, t_db

    return dot, norm_a_sq, norm_b_sq, total_elements, srank_a, srank_b


def sanitize(val: Optional[float]) -> Optional[float]:
    return None if (val is None or math.isnan(val) or math.isinf(val)) else round(val, 6)


def main() -> int:
    args = parse_args()

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

    print(
        f"[2/4] Streaming geometric features & Stable Rank estimation "
        f"(Power iters: {args.power_iters})..."
    )
    raw_stats = []
    all_norms = []

    with SliceReader(base_map) as br, SliceReader(a_map) as ar, SliceReader(b_map) as lr:
        for layer_id, keys in layers.items():
            dot = norm_a_sq = norm_b_sq = 0.0
            layer_elements = 0
            layer_srank_a_sum = 0.0
            layer_srank_b_sum = 0.0

            for key in keys:
                d, na, nb, count, sra, srb = stream_tensor_stats(
                    br.get_slice(key),
                    ar.get_slice(key),
                    lr.get_slice(key),
                    args.chunk_elements,
                    power_iters=args.power_iters,
                )
                dot += d
                norm_a_sq += na
                norm_b_sq += nb
                layer_elements += count
                layer_srank_a_sum += sra
                layer_srank_b_sum += srb

            denom = math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq)
            cosine = max(-1.0, min(1.0, dot / denom)) if denom > 0.0 else 0.0
            norm_a = math.sqrt(max(norm_a_sq, 0.0))
            norm_b = math.sqrt(max(norm_b_sq, 0.0))

            layer_d_eff = (layer_srank_a_sum + layer_srank_b_sum) / 2.0

            all_norms.extend([norm_a, norm_b])
            raw_stats.append((
                layer_id,
                cosine,
                norm_a,
                norm_b,
                layer_elements,
                layer_d_eff,
                len(keys),
            ))

    mean_norm = sum(all_norms) / len(all_norms) if all_norms else 1.0

    print("[3/4] Statistical routing & parameter calibration...")
    results = []

    for layer_id, score, norm_a, norm_b, elements, d_eff, param_count in raw_stats:
        active_dim = d_eff if args.use_stable_rank else float(elements)
        sigma_d = 1.0 / math.sqrt(active_dim) if active_dim > 0 else 1.0
        z_score = score / sigma_d

        tau_slerp = args.z_slerp * sigma_d
        tau_conflict = -args.z_conflict * sigma_d

        # 1. Routing Decision based on calibrated Z-thresholds
        if z_score >= args.z_slerp:
            cat, method = "High-Similarity", "slerp"
        elif z_score < -args.z_conflict:
            cat, method = "Conflicting", "dare_ties"
        else:
            cat, method = "Orthogonal/Moderate", "dare_ties"

        # 2. SLERP t Parameter (Fixed: Model A dominant -> t -> 0, Model B dominant -> t -> 1)
        norm_sum = norm_a + norm_b
        if norm_sum > 1e-6:
            slerp_t_raw = norm_b / norm_sum
            slerp_t = max(0.20, min(slerp_t_raw, 0.80))
        else:
            slerp_t = 0.50

        # 3. DARE-TIES Weight Parameter with Guardrails [0.15, 0.85]
        w_denom = (args.alpha_a * norm_b) + (args.alpha_b * norm_a)
        if w_denom > 1e-6:
            w_a_raw = (args.alpha_a * norm_b) / w_denom
            w_a = max(0.15, min(w_a_raw, 0.85))
            w_b = 1.0 - w_a
        else:
            w_a = w_b = 0.50

        # 4. Adaptive DARE-TIES Density Parameter
        # Penalize density progressively when z_score falls below -z_conflict
        if z_score < -args.z_conflict:
            excess_conflict = (-z_score) - args.z_conflict
            scale_range = max(1.0, args.z_conflict * 2.0)
            penalty_ratio = min(1.0, excess_conflict / scale_range)
            penalty_factor = 1.0 - (0.75 * penalty_ratio)
        else:
            penalty_factor = 1.0

        dare_density = max(0.05, min(args.base_density * penalty_factor, 0.50))

        results.append({
            "layer": layer_id,
            "cosine_similarity": score,
            "sigma_d": sigma_d,
            "z_score": z_score,
            "category": cat,
            "recommended_method": method,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "norm_ratio_a_over_b": (norm_a / norm_b) if norm_b > 0 else None,
            "slerp_t": slerp_t,
            "dare_weight_a": w_a,
            "dare_weight_b": w_b,
            "dare_density": dare_density,
            "effective_dim": d_eff,
            "physical_elements": elements,
            "parameter_tensors": param_count,
            "dynamic_tau_slerp": tau_slerp,
            "dynamic_tau_conflict": tau_conflict,
        })

    print("\n" + "=" * 135)
    dim_type = "Stable Rank (D_eff)" if args.use_stable_rank else "Physical Elements (D)"
    print(
        f" RoMM Statistical Routing Summary [{dim_type}] "
        f"(Z_slerp: +{args.z_slerp}σ, Z_conflict: -{args.z_conflict}σ, Mean ||v||: {mean_norm:.4f})"
    )
    print("=" * 135)
    print(
        f"{'Layer':<6} | {'Cosine':<11} | {'D_eff':<8} | {'Z-score':<9} | "
        f"{'Category':<19} | {'Method':<10} | {'||dA||':<7} | {'||dB||':<7} | Calibrated Parameters"
    )
    print("-" * 135)
    for r in results:
        param_str = (
            f"t={r['slerp_t']:.4f}"
            if r["recommended_method"] == "slerp"
            else f"w=[A:{r['dare_weight_a']:.3f}, B:{r['dare_weight_b']:.3f}] d={r['dare_density']:.3f}"
        )
        print(
            f"L{r['layer']:<5} | {r['cosine_similarity']:+10.6f} | {r['effective_dim']:<8.1f} | "
            f"{r['z_score']:+7.2f}σ | {r['category']:<19} | {r['recommended_method'].upper():<10} | "
            f"{r['norm_a']:<7.4f} | {r['norm_b']:<7.4f} | {param_str}"
        )
    print("=" * 135 + "\n")

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
            "use_stable_rank": args.use_stable_rank,
            "power_iters": args.power_iters,
            "z_slerp": args.z_slerp,
            "z_conflict": args.z_conflict,
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
                    "base_model": args.model_a,
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
                                "density": round(r["dare_density"], 4),
                            },
                        },
                        {
                            "model": args.model_b,
                            "layer_range": [idx, idx + 1],
                            "parameters": {
                                "weight": round(r["dare_weight_b"], 4),
                                "density": round(r["dare_density"], 4),
                            },
                        },
                    ],
                    "merge_method": "dare_ties",
                    "base_model": args.base,
                })

        # Top-level fallback uses 'linear' to safely blend non-Transformer layers
        # (embeddings, norms, lm_head) without triggering base_model missing errors.
        yaml_dict = {
            "merge_method": "linear",
            "dtype": "bfloat16",
            "parameters": {"weight": 0.5},
            "slices": slices,
        }

        try:
            import yaml
            with yaml_path.open("w", encoding="utf-8") as f:
                yaml.dump(yaml_dict, f, sort_keys=False)
        except ImportError:
            with yaml_path.open("w", encoding="utf-8") as f:
                f.write("merge_method: linear\n")
                f.write("dtype: bfloat16\n")
                f.write("parameters:\n  weight: 0.5\n")
                f.write("slices:\n")
                for s in slices:
                    f.write("- sources:\n")
                    for src in s["sources"]:
                        f.write(f"  - model: {src['model']}\n")
                        f.write(f"    layer_range:\n    - {src['layer_range'][0]}\n    - {src['layer_range'][1]}\n")
                        if "parameters" in src:
                            f.write("    parameters:\n")
                            f.write(f"      weight: {src['parameters']['weight']}\n")
                            f.write(f"      density: {src['parameters']['density']}\n")
                    f.write(f"  merge_method: {s['merge_method']}\n")
                    if "base_model" in s:
                        f.write(f"  base_model: {s['base_model']}\n")
                    if "parameters" in s:
                        f.write("  parameters:\n")
                        f.write(f"    t: {s['parameters']['t']}\n")
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