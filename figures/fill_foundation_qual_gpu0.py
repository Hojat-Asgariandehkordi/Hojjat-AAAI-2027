#!/usr/bin/env python3
"""Fill missing foundation columns (VISTA3D / SegVol / nnInteractive) on GPU 0.

Usage:
  # Phase A — ddpm1 (SegVol + VISTA3D), keeps existing Ours/SAM preds if pickle exists
  CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONPATH=/path/VAE_Conrol_NET:/path/VAE_Conrol_NET/ControlNet_Diffusion \\
    /home/morteza/.conda/envs/ddpm1/bin/python \\
      Hojjat-AAAI-2027/figures/fill_foundation_qual_gpu0.py --phase ddpm --datasets lidc,maisi

  # Phase B — nninteractive env
  CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONPATH=... \\
    /home/morteza/.conda/envs/nninteractive/bin/python \\
      Hojjat-AAAI-2027/figures/fill_foundation_qual_gpu0.py --phase nni --datasets lidc,maisi

  # Phase C — redraw figures (any env with matplotlib/torch)
  ... fill_foundation_qual_gpu0.py --phase redraw --datasets lidc,maisi
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CNET = ROOT / "ControlNet_Diffusion"
FIG = Path(__file__).resolve().parent
CACHE = FIG / "paper_qual_cache"

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / ".mpl"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CNET))

import scripts  # noqa: E402

from make_qualitative_paper_figures import (  # noqa: E402
    ABLATION_CASES,
    ABLATION_VARIANTS,
    FM_KEYS,
    LIDC_CASES,
    MAISI_CASES,
    _draw_paper_grid,
    _merge_extra_foundations,
    _ours_key,
    _patch_foundation_imports,
)

LOG = logging.getLogger("paper.qual.fill")


def _stores_to_blob(stores):
    """Plain dicts — CaseStore from pyc reload is not pickle-stable."""
    blob = []
    for s in stores:
        blob.append(
            {
                "sample_index": int(s.sample_index),
                "z": int(s.z),
                "gt_hu": s.gt_hu,
                "nodule": s.nodule,
                "preds": dict(getattr(s, "preds", {}) or {}),
                "ours_inp": dict(getattr(s, "ours_inp", {}) or {}),
            }
        )
    return blob


def _blob_to_stores(blob):
    from types import SimpleNamespace

    return [SimpleNamespace(**item) for item in blob]


def _save_stores(pkl: Path, stores) -> None:
    pkl.write_bytes(pickle.dumps(_stores_to_blob(stores)))


def _load_stores(pkl: Path):
    return _blob_to_stores(pickle.loads(pkl.read_bytes()))


def _load_or_build_stores(dataset: str, force_rebuild: bool):
    from scripts.setting import load_yaml_config, setup_logging

    logger = setup_logging(f"paper.qual.fill.{dataset}")
    if dataset == "lidc":
        cfg = load_yaml_config(CNET / "configs/benchmark_test_foundation.yaml")
        pkl = CACHE / "lidc_stores.pkl"
        cases = sorted(set(LIDC_CASES + ABLATION_CASES))
        ours_variants = ABLATION_VARIANTS
        out = CACHE / "lidc_raw"
    else:
        cfg = load_yaml_config(CNET / "configs/benchmark_test_maisi_foundation.yaml")
        pkl = CACHE / "maisi_stores.pkl"
        cases = list(MAISI_CASES)
        ours_variants = [(60, 3)]
        out = CACHE / "maisi_raw"

    if pkl.is_file() and not force_rebuild:
        LOG.info("Loading existing stores %s", pkl)
        return cfg, _load_stores(pkl), cases, pkl

    # Rebuild path needs visualize (ddpm1 / py3.9 bytecode).
    import scripts.visualize_ours_step_refine_comparison as viz

    _patch_foundation_imports()
    out.mkdir(parents=True, exist_ok=True)
    captured: dict = {}
    orig = viz.draw_variants_grid

    def _capture(stores, col_keys, col_labels, out_path, *, hu_min, hu_max):
        captured["stores"] = stores
        try:
            orig(stores, col_keys, col_labels, out_path, hu_min=hu_min, hu_max=hu_max)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("draw_variants_grid skipped (%s)", exc)

    viz.draw_variants_grid = _capture  # type: ignore
    try:
        # Builtin loop: sam_med2d + segvol + vista3d; ours for paper grids.
        viz.run_qualitative(
            cfg,
            ROOT,
            out,
            cases,
            logger,
            foundation_keys=["sam_med2d", "segvol", "vista3d"],
            ours_variants=ours_variants,
        )
    finally:
        viz.draw_variants_grid = orig

    stores = captured.get("stores")
    if not stores:
        raise RuntimeError(f"No CaseStore captured for {dataset}")
    # sam_med3d via extra merge (nnInteractive skipped in ddpm phase)
    _merge_extra_foundations(cfg, ROOT, cases, stores, logger)
    _save_stores(pkl, stores)
    LOG.info("Wrote %s (%d stores)", pkl, len(stores))
    return cfg, stores, cases, pkl


def _resolve_path(repo: Path, value) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else (repo / p).resolve()


def _run_methods_on_stores(cfg, stores, cases, methods: list[str], logger) -> None:
    """Run selected foundation pipelines and write into store.preds."""
    data_cfg = dict(cfg)
    # Prefer lightweight path resolve (avoid visualize_* imports on py3.12).
    try:
        from scripts.benchmark_test_foundation import normalize_inpaint_data_cfg

        data_cfg = normalize_inpaint_data_cfg(cfg)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("normalize_inpaint_data_cfg unavailable (%s); using raw cfg", exc)
    cache_path = _resolve_path(ROOT, data_cfg.get("positive_patch_cache", cfg.get("positive_patch_cache")))
    data_cfg["positive_patch_cache"] = str(cache_path)
    data_cfg["use_positive_patch_cache"] = True
    common_kw = dict(
        hu_min=float(cfg.get("hu_min", -1000)),
        hu_max=float(cfg.get("hu_max", 400)),
        threshold=float(cfg.get("nodule_threshold", 0.5)),
        padding_voxels=int(cfg.get("bbox_padding_voxels_fm", 1)),
        device="cuda",
        include_distance_metrics=False,
        voxel_spacing=tuple(cfg.get("target_spacing", [0.7, 0.7, 1.5])),
    )
    by_idx = {int(s.sample_index): s for s in stores}

    runners = {}
    if "vista3d" in methods:
        from scripts.benchmark_vista3d import run_vista3d_benchmark_pipeline

        runners["vista3d"] = run_vista3d_benchmark_pipeline
    if "segvol" in methods:
        from scripts.benchmark_segvol import run_segvol_benchmark_pipeline

        runners["segvol"] = run_segvol_benchmark_pipeline
    if "nninteractive" in methods:
        from scripts.benchmark_nninteractive import run_nninteractive_benchmark_pipeline

        runners["nninteractive"] = run_nninteractive_benchmark_pipeline

    import numpy as np

    for key, fn in runners.items():
        LOG.info("Running %s on %d cases (cuda=%s)", key, len(cases), os.environ.get("CUDA_VISIBLE_DEVICES"))
        try:
            results, _rows = fn(cases, data_cfg, **common_kw)
            n_ok = 0
            for case in results:
                st = by_idx.get(int(case["index"]))
                if st is None:
                    continue
                st.preds[key] = np.asarray(case["pred"], dtype=np.float32)
                n_ok += 1
            LOG.info("%s wrote preds for %d/%d cases", key, n_ok, len(cases))
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed: %s", key, exc)


def _redraw(dataset: str, cfg, stores) -> None:
    hu_min, hu_max = float(cfg.get("hu_min", -1000)), float(cfg.get("hu_max", 400))
    if dataset == "lidc":
        ours_present = []
        for st in stores:
            bag = getattr(st, "ours_inp", None)
            if isinstance(bag, dict) and bag:
                ours_present = list(bag.keys())
                break

        def _match(steps: int, rounds: int) -> str:
            want = _ours_key(steps, rounds)
            if want in ours_present:
                return want
            for k in ours_present:
                kl = str(k).lower().replace(" ", "")
                if f"{steps}s_r{rounds}" in kl or (f"{steps}-step" in str(k).lower() and f"r={rounds}" in str(k).lower()):
                    return k
            return want

        ours_k = _match(40, 2)
        col_keys = ["gt"] + FM_KEYS + [ours_k]
        col_labels = {
            "gt": "GT",
            "sam_med2d": "SAM-Med2D",
            "nninteractive": "nnInteractive",
            "sam_med3d": "SAM-Med3D",
            "segvol": "SegVol",
            "vista3d": "VISTA3D",
            ours_k: "Ours (40-step, R=2)",
        }
        main = [s for s in stores if s.sample_index in LIDC_CASES]
        main.sort(key=lambda s: LIDC_CASES.index(s.sample_index))
        _draw_paper_grid(
            main,
            col_keys,
            col_labels,
            "fig_qual_lidc",
            "LIDC — red=GT, lime=prediction",
            hu_min,
            hu_max,
        )
        # report which FM keys present
        for key in FM_KEYS:
            n = sum(1 for s in main if key in getattr(s, "preds", {}) and s.preds.get(key) is not None)
            LOG.info("LIDC column %s: %d/%d", key, n, len(main))
    else:
        ours_k = _ours_key(60, 3)
        for st in stores:
            bag = getattr(st, "ours_inp", None)
            if isinstance(bag, dict) and bag:
                ours_k = next(iter(bag.keys()))
                break
        col_keys = ["gt"] + FM_KEYS + [ours_k]
        col_labels = {
            "gt": "GT",
            "sam_med2d": "SAM-Med2D",
            "nninteractive": "nnInteractive",
            "sam_med3d": "SAM-Med3D",
            "segvol": "SegVol",
            "vista3d": "VISTA3D",
            ours_k: "Ours (60-step, R=3)",
        }
        stores = sorted(stores, key=lambda s: MAISI_CASES.index(s.sample_index))
        _draw_paper_grid(
            stores,
            col_keys,
            col_labels,
            "fig_qual_maisi",
            "MAISI — red=GT, lime=prediction",
            hu_min,
            hu_max,
        )
        for key in FM_KEYS:
            n = sum(1 for s in stores if key in getattr(s, "preds", {}) and s.preds.get(key) is not None)
            LOG.info("MAISI column %s: %d/%d", key, n, len(stores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ddpm", "nni", "redraw", "all_ddpm_then_hint"], default="ddpm")
    ap.add_argument("--datasets", default="lidc,maisi")
    ap.add_argument("--force-rebuild", action="store_true", help="Rebuild CaseStores from scratch")
    ap.add_argument("--methods", default="", help="Override methods CSV for ddpm phase")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    CACHE.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    from scripts.setting import setup_logging

    logger = setup_logging("paper.qual.fill")

    for dataset in datasets:
        if args.phase == "ddpm":
            methods = [m.strip() for m in (args.methods or "segvol,vista3d").split(",") if m.strip()]
            cfg, stores, cases, pkl = _load_or_build_stores(dataset, force_rebuild=args.force_rebuild)
            # If we just rebuilt, builtin already ran segvol+vista; still re-run requested methods
            # so a SegVol fix is applied even when pickle existed from a failed-segvol run.
            if not args.force_rebuild or set(methods) - {"sam_med2d"}:
                _patch_foundation_imports()
                _run_methods_on_stores(cfg, stores, cases, methods, logger)
                _save_stores(pkl, stores)
                LOG.info("Updated %s", pkl)
        elif args.phase == "nni":
            cfg, stores, cases, pkl = _load_or_build_stores(dataset, force_rebuild=False)
            _run_methods_on_stores(cfg, stores, cases, ["nninteractive"], logger)
            _save_stores(pkl, stores)
            LOG.info("Updated %s with nnInteractive", pkl)
        elif args.phase == "redraw":
            cfg, stores, cases, pkl = _load_or_build_stores(dataset, force_rebuild=False)
            _redraw(dataset, cfg, stores)
        else:
            raise ValueError(args.phase)

    LOG.info("Phase %s done for %s", args.phase, datasets)


if __name__ == "__main__":
    main()
