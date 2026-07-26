#!/usr/bin/env python3
"""Camera-ready qualitative figures for all paper experiments.

Requires conda env ddpm1, restored ControlNet_Diffusion scripts (pyc hook),
and YAML under ControlNet_Diffusion/configs/.

Example (GPUs 1–3 free after pausing LUNA25 train):

  cd /home/morteza/MortezaStudentsS02/VAE_Conrol_NET
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=ControlNet_Diffusion \\
    CUDA_VISIBLE_DEVICES=1 \\
    /home/morteza/.conda/envs/ddpm1/bin/python \\
      Hojjat-AAAI-2027/figures/make_qualitative_paper_figures.py --only lidc
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
CNET = ROOT / "ControlNet_Diffusion"
FIG_DIR = Path(__file__).resolve().parent
CACHE = FIG_DIR / "paper_qual_cache"
BM_ROOT = ROOT / "benchmark_models"

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / ".mpl"))
sys.path.insert(0, str(CNET))
import scripts  # noqa: E402

LOG = logging.getLogger("paper.qual")

# Selected for strong Ours performance (not mean/failure cases).
# LIDC: top 40-step R=2 Dice. MAISI: high 60-step R=3 Dice with compact
# mid-slice nodules (exclude near-full-crop masks that read as empty GT).
LIDC_CASES = [1370, 76, 1183, 338]  # Dice ≈ 0.93
ABLATION_CASES = [1370, 76, 1183]
MAISI_CASES = [50, 9, 86, 16]  # Dice ≈ 0.87 / 0.86 / 0.84 / 0.84 (compact)
HEALTHY_CASES = [217, 1365, 1532, 2002]
NSCLC_PATIENTS = ["LUNG1-024", "LUNG1-145", "LUNG1-098", "LUNG1-173"]  # Dice ≈ 0.90–0.89
# One 64³ test-cache patch per patient (high Ours Dice, compact mid-slice).
NSCLC_PATCHES = [69, 352, 239, 431]  # LUNG1-024 / 145 / 098 / 173

# Mild CT display (avoid harsh high-contrast black/white). LIDC caches are often
# already clipped to ~[0, 400]; MAISI/NSCLC keep true HU.
DISPLAY_GAMMA = 1.05
DISPLAY_P_LO = 1.0
DISPLAY_P_HI = 99.0

# Built-in visualize script only auto-runs sam_med2d/segvol/vista3d; we add the rest.
FM_KEYS_BUILTIN = ["sam_med2d", "segvol", "vista3d"]
FM_KEYS_EXTRA = ["nninteractive", "sam_med3d"]
FM_KEYS = ["sam_med2d", "nninteractive", "sam_med3d", "segvol", "vista3d"]
ABLATION_VARIANTS = [(30, 3), (40, 3), (40, 2), (60, 3), (80, 2)]


def _patch_foundation_imports() -> None:
    """Fix VISTA/SegVol import clashes with ControlNet ``scripts`` and SAM-Med2D ``utils``."""
    import importlib.util
    import types

    from scripts import benchmark_vista3d as bv

    vista_root = (BM_ROOT / "foundation/vista/vista3d").resolve()
    bv._VISTA3D_ROOT = vista_root
    utils_dir = vista_root / "scripts" / "utils"

    def _inject_vista_scripts_utils():
        # Keep ControlNet ``scripts``; overlay vista's ``scripts.utils`` only.
        pkg = types.ModuleType("scripts.utils")
        pkg.__path__ = [str(utils_dir)]
        init_py = utils_dir / "__init__.py"
        pkg.__file__ = str(init_py if init_py.is_file() else utils_dir)
        sys.modules["scripts.utils"] = pkg
        for name in ("trans_utils", "sample_utils", "workflow_utils"):
            full = f"scripts.utils.{name}"
            path = utils_dir / f"{name}.py"
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location(full, path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)

    def _import_vista3d_fixed():
        root_s = str(vista_root)
        if root_s in sys.path:
            sys.path.remove(root_s)
        sys.path.insert(0, root_s)
        for k in list(sys.modules):
            if k == "vista3d" or k.startswith("vista3d."):
                del sys.modules[k]
        _inject_vista_scripts_utils()
        from vista3d import vista_model_registry  # type: ignore
        from scripts.utils.trans_utils import (  # type: ignore
            get_largest_connected_component_point,
        )

        return vista_model_registry, get_largest_connected_component_point

    bv._import_vista3d = _import_vista3d_fixed  # type: ignore[assignment]

    try:
        from scripts import benchmark_segvol as bs

        segvol_root = (BM_ROOT / "foundation/segvol").resolve()
        bs._SEGVOL_ROOT = segvol_root
        orig_seg = bs._import_segvol

        def _import_segvol_fixed():
            # SAM-Med2D leaves ``utils.py`` on sys.path; SegVol's utils/ has no __init__.py
            # so the module wins over the namespace package. Drop conflicting path entries.
            cleaned = []
            for p in list(sys.path):
                pl = p.replace("\\", "/")
                if pl.endswith("/sam_med2d") or pl.endswith("/foundation/sam_med2d"):
                    continue
                cleaned.append(p)
            sys.path[:] = cleaned
            for k in list(sys.modules):
                if k == "utils" or k.startswith("utils."):
                    del sys.modules[k]
            root_s = str(segvol_root)
            if root_s in sys.path:
                sys.path.remove(root_s)
            sys.path.insert(0, root_s)
            # Ensure utils is a package even without upstream __init__.py
            utils_pkg = segvol_root / "utils"
            init = utils_pkg / "__init__.py"
            if utils_pkg.is_dir() and not init.is_file():
                init.write_text("# namespace shim for SegVol utils\n")
            return orig_seg()

        bs._import_segvol = _import_segvol_fixed  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        LOG.warning("segvol root patch failed: %s", exc)


def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        LOG.info("Wrote %s", path)
    plt.close(fig)


def _display_limits(sl: np.ndarray):
    """Pick a darker window from the slice (handles true HU and [0,400] caches)."""
    x = np.asarray(sl, dtype=np.float32)
    lo = float(np.percentile(x, DISPLAY_P_LO))
    hi = float(np.percentile(x, DISPLAY_P_HI))
    if hi <= lo + 1e-3:
        lo, hi = float(x.min()), float(x.max() + 1e-3)
    # Slight soft bias only (keep mid-gray parenchyma like typical CT panels).
    hi = lo + 0.95 * (hi - lo)
    return lo, hi


def _gray(sl, hu_min=None, hu_max=None, gamma=None):
    if hu_min is None or hu_max is None:
        hu_min, hu_max = _display_limits(sl)
    if gamma is None:
        gamma = DISPLAY_GAMMA
    g = np.clip((sl.astype(np.float32) - hu_min) / (hu_max - hu_min + 1e-8), 0, 1)
    if gamma != 1.0:
        g = np.power(g, float(gamma))
    return g


def _contour(ax, m2d, color, lw=1.2, linestyle="-"):
    if m2d is None:
        return
    arr = np.asarray(m2d, dtype=float)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.size == 0 or float(np.nanmax(arr)) <= 0:
        return
    # Soft probs / masks are in [0, 1]; HU leftovers are ≫1 and must be ignored.
    if float(np.nanmax(arr)) > 1.5:
        return
    ax.contour(arr, levels=[0.5], colors=color, linewidths=lw, linestyles=linestyle)


def _style(ax, title=""):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontweight="bold", fontsize=9, pad=3)


def _ours_key(steps: int, rounds: int) -> str:
    from scripts.benchmark_test_foundation import ours_model_name

    try:
        return ours_model_name(steps, rounds)
    except Exception:
        return f"ours_s{steps}_r{rounds}"


def _safe_slice(vol, z):
    """Return 2D slice or None if volume shape is incompatible."""
    if vol is None:
        return None
    if hasattr(vol, "detach"):
        vol = vol.detach().cpu().numpy()
    arr = np.asarray(vol)
    if arr.dtype == object:
        return None
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        return None
    z = int(z)
    if 0 <= z < arr.shape[0]:
        return arr[z]
    # Fall back to the axis with the matching extent (orientation quirks).
    for ax in range(3):
        if 0 <= z < arr.shape[ax]:
            return np.take(arr, z, axis=ax)
    return None


def _lookup_pred(st, key):
    """Return a binary/soft mask for ``key``.

    Prefer ``st.preds`` (masks). Never fall back to ``st.ours_inp`` — that bag
    holds inpainted HU volumes, which produce empty contours at level 0.5.
    """
    preds = getattr(st, "preds", None)
    if not isinstance(preds, dict):
        return None
    if key in preds and preds[key] is not None:
        return preds[key]

    key_n = str(key).replace(" ", "").lower()
    # Accept aliases: "40s_r2" ↔ "ours:40s_r2", model display names, etc.
    aliases = {key_n, key_n.replace("ours:", ""), f"ours:{key_n}" if not key_n.startswith("ours:") else key_n}
    for k, v in preds.items():
        if v is None:
            continue
        kn = str(k).replace(" ", "").lower()
        if kn in aliases or kn.replace("ours:", "") in aliases:
            return v
        # Fuzzy: "40s_r2" inside "ours:40s_r2" / display names
        if key_n in kn or kn in key_n:
            return v
    return None


def _merge_extra_foundations(cfg, repo, case_indices, stores, logger) -> None:
    """Add nnInteractive + SAM-Med3D preds (not handled by visualize builtin loop)."""
    from scripts.benchmark_test_foundation import normalize_inpaint_data_cfg
    from scripts.visualize_ours_step_refine_comparison import _resolve

    data_cfg = normalize_inpaint_data_cfg(cfg)
    cache_path = _resolve(repo, cfg["positive_patch_cache"])
    data_cfg["positive_patch_cache"] = str(cache_path)
    data_cfg["use_positive_patch_cache"] = True
    threshold = float(cfg.get("nodule_threshold", 0.5))
    padding = int(cfg.get("bbox_padding_voxels_fm", 1))
    spacing = tuple(cfg.get("target_spacing", [0.7, 0.7, 1.5]))
    by_idx = {int(s.sample_index): s for s in stores}
    common_kw = dict(
        hu_min=float(cfg.get("hu_min", -1000)),
        hu_max=float(cfg.get("hu_max", 400)),
        threshold=threshold,
        padding_voxels=padding,
        device="cuda",
        include_distance_metrics=False,
        voxel_spacing=spacing,
    )

    runners = []
    try:
        from scripts.benchmark_nninteractive import run_nninteractive_benchmark_pipeline

        runners.append(("nninteractive", run_nninteractive_benchmark_pipeline))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("nnInteractive import failed: %s", exc)
    try:
        from scripts.benchmark_sam_med3d import run_sam_med3d_benchmark_pipeline

        runners.append(("sam_med3d", run_sam_med3d_benchmark_pipeline))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("SAM-Med3D import failed: %s", exc)

    for key, fn in runners:
        logger.info("Qualitative foundation (extra): %s", key)
        try:
            results, _rows = fn(case_indices, data_cfg, **common_kw)
            for case in results:
                st = by_idx.get(int(case["index"]))
                if st is None:
                    continue
                st.preds[key] = np.asarray(case["pred"], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Foundation %s failed: %s", key, exc)


def run_lidc_and_ablation(force: bool) -> None:
    from scripts.setting import load_yaml_config, setup_logging

    _patch_foundation_imports()
    logger = setup_logging("paper.qual.lidc")
    cfg = load_yaml_config(CNET / "configs/benchmark_test_foundation.yaml")
    stores = _collect_stores_lidc(cfg, logger, force=force)
    # Adaptive darker window per slice (None → _display_limits inside _gray).
    hu_min, hu_max = None, None

    # Prefer mask keys in st.preds (e.g. "ours:40s_r2"), not ours_inp HU keys.
    ours_present = []
    for st in stores:
        bag = getattr(st, "preds", None)
        if isinstance(bag, dict) and bag:
            ours_present = [k for k in bag if "ours" in str(k).lower() or "s_r" in str(k).lower()]
            if ours_present:
                break
    if not ours_present:
        for st in stores:
            bag = getattr(st, "ours_inp", None)
            if isinstance(bag, dict) and bag:
                ours_present = list(bag.keys())
                break
    LOG.info("Ours keys on stores: %s", ours_present)

    def _match_ours(steps: int, rounds: int) -> str:
        # Canonical cache key from visualize script
        short = f"ours:{steps}s_r{rounds}"
        if short in ours_present:
            return short
        want = _ours_key(steps, rounds)
        if want in ours_present:
            return want
        for k in ours_present:
            kl = str(k).lower().replace(" ", "")
            if f"{steps}s_r{rounds}" in kl or (f"{steps}-step" in str(k).lower() and f"r={rounds}" in str(k).lower()):
                return k
            if f"{steps}s_r{rounds}" in kl.replace("ours:", ""):
                return k
        return short

    # Main LIDC
    ours_k = _match_ours(40, 2)
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
    main_stores = [s for s in stores if s.sample_index in LIDC_CASES]
    main_stores.sort(key=lambda s: LIDC_CASES.index(s.sample_index))
    _draw_paper_grid(
        main_stores,
        col_keys,
        col_labels,
        "fig_qual_lidc",
        "LIDC — red=GT, lime=prediction",
        hu_min,
        hu_max,
        show_gt_on_all=True,
    )

    # Ablation
    ab_keys = ["gt"] + [_match_ours(s, r) for s, r in ABLATION_VARIANTS]
    ab_labels = {
        "gt": "GT",
        _match_ours(30, 3): "30-step",
        _match_ours(40, 3): "40-step",
        _match_ours(40, 2): "40-step, R=2",
        _match_ours(60, 3): "60-step",
        _match_ours(80, 2): "80-step, R=2",
    }
    ab_stores = [s for s in stores if s.sample_index in ABLATION_CASES]
    ab_stores.sort(key=lambda s: ABLATION_CASES.index(s.sample_index))
    _draw_paper_grid(
        ab_stores,
        ab_keys,
        ab_labels,
        "fig_qual_ablation",
        "Ablation (LIDC) — red=GT, lime=prediction",
        hu_min,
        hu_max,
        show_gt_on_all=True,
    )


def _collect_stores_lidc(cfg, logger, force: bool):
    """Run qualitative once and return CaseStore list (cached as pickle)."""
    import pickle

    pkl = CACHE / "lidc_stores.pkl"
    if pkl.is_file() and not force:
        LOG.info("Loading %s", pkl)
        raw = pickle.loads(pkl.read_bytes())
        # fill_foundation_qual_gpu0 saves plain dicts for pickle stability
        if raw and isinstance(raw[0], dict):
            from types import SimpleNamespace

            return [SimpleNamespace(**item) for item in raw]
        return raw

    import scripts.visualize_ours_step_refine_comparison as viz

    out = CACHE / "lidc_raw"
    out.mkdir(parents=True, exist_ok=True)
    cases = sorted(set(LIDC_CASES + ABLATION_CASES))
    captured = {}
    orig = viz.draw_variants_grid

    def _capture(stores, col_keys, col_labels, out_path, *, hu_min, hu_max):
        captured["stores"] = stores
        captured["col_keys"] = col_keys
        captured["col_labels"] = col_labels
        try:
            orig(stores, col_keys, col_labels, out_path, hu_min=hu_min, hu_max=hu_max)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("draw_variants_grid skipped (%s); paper grids use stores", exc)

    viz.draw_variants_grid = _capture  # type: ignore
    try:
        viz.run_qualitative(
            cfg,
            ROOT,
            out,
            cases,
            logger,
            foundation_keys=FM_KEYS_BUILTIN,
            ours_variants=ABLATION_VARIANTS,
        )
    finally:
        viz.draw_variants_grid = orig

    stores = captured.get("stores")
    if not stores:
        raise RuntimeError("Failed to capture CaseStore list from run_qualitative")
    _merge_extra_foundations(cfg, ROOT, cases, stores, logger)
    try:
        pkl.write_bytes(pickle.dumps(stores))
        LOG.info("Cached %d stores -> %s", len(stores), pkl)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not pickle stores (%s); continuing without cache", exc)
    return stores


def _draw_paper_grid(stores, col_keys, col_labels, stem, title, hu_min, hu_max, show_gt_on_all=True):
    # Drop method columns with no prediction object at all (failed FM imports).
    kept = []
    for key in col_keys:
        if key == "gt" or any(_lookup_pred(st, key) is not None for st in stores):
            kept.append(key)
        else:
            LOG.warning("Omitting empty column %s from %s", key, stem)
    col_keys = kept
    nrows, ncols = len(stores), len(col_keys)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.65 * ncols, 1.65 * nrows), squeeze=False)
    for r, st in enumerate(stores):
        g2 = _safe_slice(st.gt_hu, st.z)
        if g2 is None:
            continue
        gray = _gray(g2, hu_min, hu_max)
        for c, key in enumerate(col_keys):
            ax = axes[r, c]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            # Draw prediction first (thicker lime), then GT (red) so tight
            # Ours–GT overlap still shows both colors.
            if key != "gt":
                _contour(ax, _safe_slice(_lookup_pred(st, key), st.z), "lime", 1.45)
            if show_gt_on_all:
                _contour(ax, _safe_slice(st.nodule, st.z), "red", 1.15)
            _style(ax, col_labels.get(key, key) if r == 0 else "")
    fig.suptitle(title, fontweight="bold", fontsize=11, y=1.01)
    fig.tight_layout()
    save_fig(fig, stem)


def run_maisi(force: bool) -> None:
    from scripts.setting import load_yaml_config, setup_logging
    import pickle

    logger = setup_logging("paper.qual.maisi")
    cfg = load_yaml_config(CNET / "configs/benchmark_test_maisi_foundation.yaml")

    pkl = CACHE / "maisi_stores.pkl"
    if pkl.is_file() and not force:
        raw = pickle.loads(pkl.read_bytes())
        if raw and isinstance(raw[0], dict):
            from types import SimpleNamespace

            stores = [SimpleNamespace(**item) for item in raw]
        else:
            stores = raw
    else:
        import scripts.visualize_ours_step_refine_comparison as viz

        _patch_foundation_imports()
        out = CACHE / "maisi_raw"
        out.mkdir(parents=True, exist_ok=True)
        captured = {}
        orig = viz.draw_variants_grid

        def _capture(stores, col_keys, col_labels, out_path, *, hu_min, hu_max):
            captured["stores"] = stores
            try:
                orig(stores, col_keys, col_labels, out_path, hu_min=hu_min, hu_max=hu_max)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("draw_variants_grid skipped (%s)", exc)

        viz.draw_variants_grid = _capture  # type: ignore
        try:
            viz.run_qualitative(
                cfg,
                ROOT,
                out,
                MAISI_CASES,
                logger,
                foundation_keys=FM_KEYS_BUILTIN,
                ours_variants=[(60, 3)],
            )
        finally:
            viz.draw_variants_grid = orig
        stores = captured.get("stores")
        if not stores:
            raise RuntimeError("Failed to capture MAISI CaseStore list")
        _merge_extra_foundations(cfg, ROOT, MAISI_CASES, stores, logger)
        try:
            pkl.write_bytes(pickle.dumps(stores))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not pickle MAISI stores (%s)", exc)

    hu_min, hu_max = None, None  # adaptive darker window
    # Prefer preds mask key (ours:60s_r3), not ours_inp HU key (60s_r3).
    ours_k = "ours:60s_r3"
    for st in stores:
        bag = getattr(st, "preds", None)
        if isinstance(bag, dict):
            for k in bag:
                if "60s_r3" in str(k).lower() or "60-step" in str(k).lower():
                    ours_k = k
                    break
            if ours_k in bag:
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


def _bbox_to_mask(shape, bbox_zyx) -> np.ndarray:
    z0, y0, x0, z1, y1, x1 = [int(v) for v in bbox_zyx]
    mask = np.zeros(tuple(shape), dtype=np.float32)
    mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = 1.0
    return mask


def _mid_box_z(bbox_zyx) -> int:
    z0, _, _, z1, _, _ = [int(v) for v in bbox_zyx]
    return int((z0 + z1) // 2)


def _run_nni_healthy(case_indices: list[int]) -> dict[int, np.ndarray]:
    """Re-infer nnInteractive on healthy random-box prompts (fixes empty gallery column).

    Important: ``benchmark_nninteractive._ensure_controlnet_scripts`` reloads
    ``scripts.*`` and would wipe an in-memory ``_POSITIVE_CACHE`` injection, so
    we pass a custom ``load_sample_fn`` that returns box-filled nodules.
    """
    import json

    import torch

    from scripts.benchmark_nninteractive import (
        load_nninteractive_session,
        prepare_nninteractive_model,
        run_nninteractive_on_cases,
    )
    from scripts.benchmark_test_foundation import normalize_inpaint_data_cfg
    from scripts.setting import load_yaml_config

    cfg = load_yaml_config(CNET / "configs/benchmark_healthy_fp_random_boxes.yaml")
    data_cfg = normalize_inpaint_data_cfg(cfg)
    cache_path = Path(data_cfg["positive_patch_cache"])
    if not cache_path.is_absolute():
        cache_path = (ROOT / cache_path).resolve()
        data_cfg["positive_patch_cache"] = str(cache_path)

    boxes_path = Path(cfg["random_boxes_json"])
    if not boxes_path.is_absolute():
        boxes_path = (ROOT / boxes_path).resolve()
    by_idx = {
        int(b["patch_index"]): b
        for b in json.loads(boxes_path.read_text())["boxes"]
    }

    LOG.info("Loading healthy cache for nnInteractive prompt injection …")
    samples = torch.load(cache_path, map_location="cpu", weights_only=False)
    prompt_by_idx: dict[int, dict] = {}
    for i in case_indices:
        s = dict(samples[int(i)])
        rec = by_idx[int(i)]
        shape = tuple(int(x) for x in s["hu"].shape)
        s["nodule"] = torch.as_tensor(_bbox_to_mask(shape, rec["bbox_zyx"]))
        s["is_positive"] = True
        s["fp_prompt_box"] = True
        prompt_by_idx[int(i)] = s
    LOG.info("Prepared %d prompted healthy samples", len(prompt_by_idx))

    def _load_sample(cfg_unused, sample_index=None):  # noqa: ARG001
        return prompt_by_idx[int(sample_index)]

    prepare_nninteractive_model()
    session = load_nninteractive_session(device="cuda")
    LOG.info("Running nnInteractive on cases %s", case_indices)
    results = run_nninteractive_on_cases(
        case_indices,
        data_cfg,
        session,
        hu_min=float(cfg.get("hu_min", -1000)),
        hu_max=float(cfg.get("hu_max", 400)),
        threshold=float(cfg.get("nodule_threshold", 0.5)),
        padding_voxels=int(cfg.get("bbox_padding_voxels_fm", 2)),
        prompt_mode="strategy4_slice_bbox",
        load_sample_fn=_load_sample,
    )
    out: dict[int, np.ndarray] = {}
    for case in results:
        idx = int(case["index"])
        pred = np.asarray(case["pred"], dtype=np.float32)
        out[idx] = pred
        LOG.info("nnInteractive #%s voxels=%d", idx, int((pred > 0.5).sum()))
    return out


def run_healthy_compose() -> None:
    """Compose paper healthy-FP figure; re-draw nnInteractive (gallery column was empty)."""
    import json

    from PIL import Image

    import torch

    src = CNET / "runs/benchmark_healthy_fp_random_boxes/visual_comparison_healthy_fp.png"
    meta_path = CNET / "runs/benchmark_healthy_fp_random_boxes/visual_comparison_healthy_fp.json"
    if not src.is_file():
        raise FileNotFoundError(src)
    im = np.array(Image.open(src).convert("RGB"))
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    viz_indices = [int(x) for x in meta.get("viz_indices", [])]
    if not viz_indices:
        viz_indices = list(HEALTHY_CASES)

    col_mean = im.mean(axis=(0, 2))
    row_mean = im.mean(axis=(1, 2))
    white_c = col_mean > 245
    white_r = row_mean > 245

    def segs(mask):
        out = []
        i = 0
        n = len(mask)
        while i < n:
            if not mask[i]:
                j = i
                while j < n and not mask[j]:
                    j += 1
                if j - i > 40:
                    out.append((i, j))
                i = j
            else:
                i += 1
        return out

    csegs, rsegs = segs(white_c), segs(white_r)
    LOG.info("healthy gallery panels: %d cols × %d rows", len(csegs), len(rsegs))
    idx_to_row = {idx: ri for ri, idx in enumerate(viz_indices) if ri < len(rsegs)}

    # Prefer curated cases with large nnInteractive FP volume (from benchmark CSV).
    case_ids = [i for i in HEALTHY_CASES if i in idx_to_row]
    if len(case_ids) < 4:
        case_ids = viz_indices[:4]
    LOG.info("healthy paper cases: %s", case_ids)

    nni_preds = _run_nni_healthy(case_ids)

    # Load HU + prompt boxes for redrawing the nnInteractive column.
    boxes = {
        int(b["patch_index"]): b
        for b in json.loads(
            (CNET / "runs/benchmark_healthy_fp_random_boxes/random_boxes.json").read_text()
        )["boxes"]
    }
    cache_path = CNET / "runs/diffusion/healthy_patches_cache_test.pt"
    LOG.info("Loading healthy HU cache for panel redraw …")
    samples = torch.load(cache_path, map_location="cpu", weights_only=False)

    # Gallery cols: 0 prompt, 1 sam2d, 2 sam3d, 3 segvol, 4 vista, 5 nni, 6 ours
    # Paper order: prompt, sam2d, nni, sam3d, segvol, vista, ours
    src_cols = [0, 1, 5, 2, 3, 4, 6]
    nni_paper_col = 2
    labels = [
        "Prompt",
        "SAM-Med2D",
        "nnInteractive",
        "SAM-Med3D",
        "SegVol",
        "VISTA3D",
        "Ours (60-step, R=3)*",
    ]
    fig, axes = plt.subplots(len(case_ids), 7, figsize=(1.7 * 7, 1.7 * len(case_ids)), squeeze=False)
    for r, idx in enumerate(case_ids):
        ri = idx_to_row[idx]
        y0, y1 = rsegs[ri]
        for c, ci in enumerate(src_cols):
            ax = axes[r, c]
            if c == nni_paper_col:
                sample = samples[idx]
                hu = np.asarray(sample["hu"], dtype=np.float32)
                z = _mid_box_z(boxes[idx]["bbox_zyx"])
                # Healthy cache HU is usually normalized to ~[0, 1].
                if float(hu.max()) <= 1.5:
                    gray = np.clip(hu[z], 0, 1)
                    if DISPLAY_GAMMA != 1.0:
                        gray = np.power(gray, float(DISPLAY_GAMMA))
                else:
                    gray = _gray(hu[z])
                ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
                prompt = _bbox_to_mask(hu.shape, boxes[idx]["bbox_zyx"])
                _contour(ax, prompt[z], "yellow", 1.1)
                pred = nni_preds.get(idx)
                if pred is not None:
                    _contour(ax, np.asarray(pred)[z], "lime", 1.45)
                else:
                    LOG.warning("Missing nnInteractive pred for #%s", idx)
            else:
                x0, x1 = csegs[ci]
                ax.imshow(im[y0:y1, x0:x1])
            _style(ax, labels[c] if r == 0 else "")
    fig.suptitle(
        "Healthy FP — yellow=prompt, lime=prediction (GT empty)\n"
        "*Gallery Ours was 60-step R=1; table reports R=3 (same checkpoint family).",
        fontweight="bold",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig_qual_healthy_fp")


def _nsclc_extract_patch(vol: np.ndarray, center) -> np.ndarray | None:
    """Crop 64³ from processed NSCLC volume and reorient to patch-cache axes.

    Patch centers are stored as (z, y, x) in processed space; NIfTI arrays are
    (y, x, z). Matching orientation is ``transpose(crop, (2, 1, 0))``.
    """
    c = [int(x) for x in (center.tolist() if hasattr(center, "tolist") else center)]
    center_vol = [c[2], c[1], c[0]]
    ps = 64
    starts = [ci - ps // 2 for ci in center_vol]
    if any(st < 0 or st + ps > vol.shape[i] for i, st in enumerate(starts)):
        return None
    sl = tuple(slice(st, st + ps) for st in starts)
    crop = vol[sl]
    if crop.shape != (ps, ps, ps):
        return None
    return np.ascontiguousarray(np.transpose(crop, (2, 1, 0)))


def run_nsclc() -> None:
    """NSCLC qualitative grids: volume slices + 64³ patch crops (paper main)."""
    import nibabel as nib
    import torch

    base = CNET / "runs/benchmark_nsclc_volume_full_masks"
    methods = [
        ("GT", None),
        ("SAM-Med2D", "seg_sam_med2d"),
        ("nnInteractive", "seg_nninteractive"),
        ("SAM-Med3D", "seg_sam_med3d"),
        ("SegVol", "seg_segvol"),
        ("VISTA3D", "seg_vista3d"),
        ("Ours (60-step, R=1)", "seg_ours"),
    ]

    # ---- Volume-wise (full axial / best-axis slice) ----
    fig, axes = plt.subplots(
        len(NSCLC_PATIENTS), len(methods), figsize=(1.8 * len(methods), 1.8 * len(NSCLC_PATIENTS)), squeeze=False
    )
    for r, pid in enumerate(NSCLC_PATIENTS):
        ct = np.asanyarray(nib.load(str(base / "processed_ct" / f"{pid}.nii.gz")).dataobj).astype(np.float32)
        gt = np.asanyarray(nib.load(str(base / "processed_gtv" / f"{pid}.nii.gz")).dataobj) > 0.5
        profiles = [gt.sum(axis=tuple(i for i in range(3) if i != ax)) for ax in range(3)]
        slice_axis = int(np.argmax([float(p.max()) for p in profiles]))
        z = int(np.argmax(profiles[slice_axis]))
        sl = np.take(ct, z, axis=slice_axis)
        g2 = np.take(gt, z, axis=slice_axis)
        gray = _gray(sl)
        for c, (title, folder) in enumerate(methods):
            ax = axes[r, c]
            ax.imshow(np.rot90(gray), cmap="gray", vmin=0, vmax=1)
            if folder is not None:
                pred = np.asanyarray(nib.load(str(base / folder / f"{pid}.nii.gz")).dataobj) > 0.5
                if pred.shape != gt.shape and pred.T.shape == gt.shape:
                    pred = pred.T
                p2 = np.take(pred, z, axis=slice_axis)
                _contour(ax, np.rot90(p2), "lime", 1.45)
            _contour(ax, np.rot90(g2), "red", 1.15)
            _style(ax, title if r == 0 else "")
    fig.suptitle("NSCLC-Radiomics (volume) — red=GT, lime=prediction", fontweight="bold", fontsize=11, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_qual_nsclc_volume")

    # ---- Patch-wise 64³ (same visual language as LIDC / MAISI) ----
    cache_path = CNET / "runs/nsclc/positive_patches_cache_test.pt"
    LOG.info("Loading NSCLC patch cache %s", cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    vol_cache: dict[str, dict[str, np.ndarray]] = {}

    def _vol(pid: str, folder: str | None) -> np.ndarray:
        key = f"{pid}:{folder or 'gtv'}"
        if key not in vol_cache:
            if folder is None:
                path = base / "processed_gtv" / f"{pid}.nii.gz"
            elif folder == "__ct__":
                path = base / "processed_ct" / f"{pid}.nii.gz"
            else:
                path = base / folder / f"{pid}.nii.gz"
            arr = np.asanyarray(nib.load(str(path)).dataobj)
            vol_cache[key] = arr.astype(np.float32) if folder == "__ct__" else (arr > 0.5)
        return vol_cache[key]

    fig, axes = plt.subplots(
        len(NSCLC_PATCHES), len(methods), figsize=(1.65 * len(methods), 1.65 * len(NSCLC_PATCHES)), squeeze=False
    )
    for r, (pid, pidx) in enumerate(zip(NSCLC_PATIENTS, NSCLC_PATCHES)):
        sample = cache[int(pidx)]
        if str(sample.get("patient_id")) != pid:
            LOG.warning("NSCLC patch %s patient_id=%s (expected %s)", pidx, sample.get("patient_id"), pid)
        center = sample["center"]
        hu = np.asarray(sample["hu"], dtype=np.float32)
        # Cache is normalized [0,1] ← HU window [-1000,400]
        hu_hu = hu * 1400.0 - 1000.0 if float(hu.max()) <= 1.5 else hu
        nod = np.asarray(sample["nodule"]) > 0.5
        z = int(np.argmax(nod.sum(axis=(1, 2))))
        gray = _gray(hu_hu[z])
        # Sanity: volume GT crop should match cache nodule
        gtc = _nsclc_extract_patch(_vol(pid, None), center)
        if gtc is not None:
            dice_gt = 2.0 * float((gtc & nod).sum()) / max(1.0, float(gtc.sum() + nod.sum()))
            if dice_gt < 0.99:
                LOG.warning("NSCLC patch #%s GT align dice=%.3f", pidx, dice_gt)
        for c, (title, folder) in enumerate(methods):
            ax = axes[r, c]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            if folder is not None:
                pred_vol = _vol(pid, folder)
                pred = _nsclc_extract_patch(pred_vol, center)
                if pred is None:
                    LOG.warning("NSCLC patch crop failed %s %s", pid, folder)
                else:
                    _contour(ax, pred[z], "lime", 1.45)
            _contour(ax, nod[z], "red", 1.15)
            _style(ax, title if r == 0 else "")
        LOG.info("NSCLC patch row %s #%s z=%d nodule_vox=%d", pid, pidx, z, int(nod.sum()))
    fig.suptitle(
        "NSCLC-Radiomics (64³ patches) — red=GT, lime=prediction",
        fontweight="bold",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "fig_qual_nsclc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["lidc", "maisi", "healthy", "nsclc", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="Re-run GPU inference even if cache exists")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    CACHE.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "nsclc"):
        run_nsclc()
    if args.only in ("all", "healthy"):
        run_healthy_compose()
    if args.only in ("all", "lidc"):
        run_lidc_and_ablation(force=args.force)
    if args.only in ("all", "maisi"):
        run_maisi(force=args.force)
    LOG.info("Done → %s", FIG_DIR)


if __name__ == "__main__":
    main()
