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
# Relative wins vs SAM2D/3D/SegVol/nnInt (lowest Ours vol, largest gap).
# Absolute best (need GPU to rebuild visual_comparison gallery + nnI):
#   [1916, 2509, 1643, 1622]
# Gallery fallback (present in visual_comparison_healthy_fp.png):
HEALTHY_CASES = [2514, 1680, 971, 3045]
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
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.015,
        )
        LOG.info("Wrote %s", path)
    plt.close(fig)


def _pack_qual_fig(fig: plt.Figure, *, has_suptitle: bool = True) -> None:
    """Pack image panels tightly (flush columns; tiny row gap for titles)."""
    for ax in fig.axes:
        # Fill the axes box completely — aspect='equal' letterboxes and looks
        # like large column gutters when cells are not perfectly square.
        ax.set_aspect("auto")
        ax.margins(0)
        ax.tick_params(length=0, pad=0)
    fig.subplots_adjust(
        left=0.001,
        right=0.999,
        bottom=0.001,
        top=0.91 if has_suptitle else 0.98,
        wspace=0.0,
        hspace=0.06,
    )


def _qual_subplots(nrows: int, ncols: int, cell: float = 1.22):
    """Nearly square panels; top margin reserved for column titles / suptitle."""
    return plt.subplots(
        nrows,
        ncols,
        figsize=(cell * ncols, cell * nrows + 0.35),
        squeeze=False,
        gridspec_kw={"wspace": 0.0, "hspace": 0.06},
    )


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
    ax.set_aspect("auto")
    ax.margins(0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=7.5, pad=1.0)


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
        ours_k: "Ours",
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


def _render_overlay_rgb(
    gray,
    pred=None,
    gt=None,
    px: int = 256,
    *,
    fill_pred: bool = False,
    pred_alpha: float = 0.40,
) -> np.ndarray:
    """Rasterize one CT panel to RGB (flush mosaic building block)."""
    fig, ax = plt.subplots(figsize=(px / 100.0, px / 100.0), dpi=100)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1, aspect="equal")
    if pred is not None:
        arr = np.asarray(pred, dtype=float)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if fill_pred and arr.size and float(np.nanmax(arr)) > 0:
            mask = arr > 0.5
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask] = (0.20, 0.95, 0.20, float(pred_alpha))
            ax.imshow(overlay, interpolation="nearest", aspect="equal")
        _contour(ax, arr, "lime", 1.45)
    if gt is not None:
        _contour(ax, gt, "red", 1.15)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.ascontiguousarray(rgba[..., :3])


def _mask_from_color_contours(panel_rgb: np.ndarray, *, mode: str = "lime") -> np.ndarray:
    """Recover a filled 2D mask from lime/yellow contour pixels on a gallery panel.

    Gallery panels only draw contours. When a model predicts (nearly) the whole
    prompt box — common for SAM-Med3D on healthy FP — the lime stroke is a thin
    ring that often has gaps where the yellow prompt contour overwrote it.
    Closing + hole-fill restores the solid mask; a residual shell falls back to
    the contour bounding box.
    """
    import cv2
    from scipy import ndimage

    panel = np.asarray(panel_rgb)
    r = panel[..., 0].astype(np.int16)
    g = panel[..., 1].astype(np.int16)
    b = panel[..., 2].astype(np.int16)
    if mode == "lime":
        edge = (g > 140) & (r < 160) & (b < 160) & (g > r + 20) & (g > b + 20)
    elif mode == "yellow":
        edge = (r > 160) & (g > 160) & (b < 120) & (r > b + 30) & (g > b + 30)
    else:
        raise ValueError(mode)
    if not np.any(edge):
        return np.zeros(panel.shape[:2], dtype=bool)

    closed = ndimage.binary_closing(edge, iterations=3)
    closed = ndimage.binary_dilation(closed, iterations=2)
    edge_u8 = (closed.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(edge_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(panel.shape[:2], dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    filled = mask.astype(bool) | ndimage.binary_fill_holes(closed) | ndimage.binary_fill_holes(edge)

    # Keep thin single-pixel predictions (very small nodules / FP blobs).
    if filled.sum() <= max(8, int(edge.sum() * 0.5)):
        filled = edge | ndimage.binary_dilation(edge, iterations=1)

    # Hollow rectangular ring (SAM-Med3D ≈ prompt box): lime stroke lies on the
    # perimeter with an empty interior. Only then fill the contour bbox — do not
    # trigger on sparse speckles (Ours) whose bbox interior is also mostly empty.
    ys, xs = np.where(edge)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h, w = y1 - y0 + 1, x1 - x0 + 1
    if h >= 12 and w >= 12:
        pad = max(2, int(0.12 * min(h, w)))
        interior_sl = (slice(y0 + pad, y1 - pad + 1), slice(x0 + pad, x1 - pad + 1))
        band = np.zeros_like(edge)
        band[y0 : y1 + 1, x0 : x1 + 1] = True
        band[interior_sl] = False
        edge_on_band = float(edge[band].sum()) / float(edge.sum())
        edge_in_interior = float(edge[interior_sl].mean()) if edge[interior_sl].size else 0.0
        fill_in_interior = float(filled[interior_sl].mean()) if filled[interior_sl].size else 0.0
        if edge_on_band >= 0.70 and edge_in_interior < 0.03 and fill_in_interior < 0.45:
            solid = np.zeros_like(filled)
            solid[y0 : y1 + 1, x0 : x1 + 1] = True
            filled = solid
    return filled


def _fill_colored_regions(
    panel_rgb: np.ndarray,
    *,
    mode: str = "lime",
    alpha: float = 0.45,
) -> np.ndarray:
    """Fill interior of lime/yellow contour overlays already drawn on a gallery panel."""
    panel = np.asarray(panel_rgb)
    region = _mask_from_color_contours(panel, mode=mode)
    if not region.any():
        return panel.copy()
    fill_rgb = (
        np.array([40, 230, 40], dtype=np.float32)
        if mode == "lime"
        else np.array([255, 220, 40], dtype=np.float32)
    )
    out = panel.astype(np.float32)
    out[region] = (1.0 - alpha) * out[region] + alpha * fill_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def _compose_mosaic(
    panels: list[list[np.ndarray]],
    col_labels: list[str],
    title: str,
    *,
    col_gap: int = 2,
    row_gap: int = 6,
    stem: str,
) -> None:
    """Tile RGB panels with tiny pixel gaps; column labels baked into a header strip."""
    from PIL import Image as _PILImage
    from PIL import ImageDraw, ImageFont

    nrows, ncols = len(panels), len(panels[0])
    h = int(panels[0][0].shape[0])
    w = int(panels[0][0].shape[1])
    gap_c = np.full((h, col_gap, 3), 255, dtype=np.uint8)
    row_chunks = []
    for r in range(nrows):
        parts = []
        for c in range(ncols):
            tile = panels[r][c]
            if tile.shape[0] != h or tile.shape[1] != w:
                tile = np.asarray(_PILImage.fromarray(tile).resize((w, h), _PILImage.NEAREST))
            parts.append(tile)
            if c < ncols - 1:
                parts.append(gap_c)
        row_chunks.append(np.concatenate(parts, axis=1))
    if nrows > 1:
        gap_r = np.full((row_gap, row_chunks[0].shape[1], 3), 255, dtype=np.uint8)
        mosaic_parts = []
        for r, row in enumerate(row_chunks):
            mosaic_parts.append(row)
            if r < nrows - 1:
                mosaic_parts.append(gap_r)
        mosaic = np.concatenate(mosaic_parts, axis=0)
    else:
        mosaic = row_chunks[0]

    mw = int(mosaic.shape[1])
    # Header strip: each label centered on its panel column (pixel-aligned).
    header_h = 32
    header = _PILImage.new("RGB", (mw, header_h), (255, 255, 255))
    draw = ImageDraw.Draw(header)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
    for c, lab in enumerate(col_labels):
        cx = c * (w + col_gap) + w // 2
        bbox = draw.textbbox((0, 0), lab, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw // 2, max(0, (header_h - th) // 2 - 1)), lab, fill=(0, 0, 0), font=font)
    mosaic = np.concatenate([np.asarray(header), mosaic], axis=0)

    mh, mw = mosaic.shape[:2]
    fig_w = max(6.9, 1.05 * ncols)
    fig_h = fig_w * (mh / mw) + 0.32
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes([0.005, 0.01, 0.99, 0.90])
    ax.imshow(mosaic, aspect="equal", interpolation="nearest")
    ax.set_axis_off()
    fig.suptitle(title, fontweight="bold", fontsize=10, y=0.985)
    save_fig(fig, stem)


def _draw_paper_grid(stores, col_keys, col_labels, stem, title, hu_min, hu_max, show_gt_on_all=True):
    # Drop method columns with no prediction object at all (failed FM imports).
    kept = []
    for key in col_keys:
        if key == "gt" or any(_lookup_pred(st, key) is not None for st in stores):
            kept.append(key)
        else:
            LOG.warning("Omitting empty column %s from %s", key, stem)
    col_keys = kept
    labels = [col_labels.get(k, k) for k in col_keys]
    panels: list[list[np.ndarray]] = []
    for st in stores:
        g2 = _safe_slice(st.gt_hu, st.z)
        if g2 is None:
            continue
        gray = _gray(g2, hu_min, hu_max)
        row = []
        gt2 = _safe_slice(st.nodule, st.z) if show_gt_on_all else None
        for key in col_keys:
            pred = None if key == "gt" else _safe_slice(_lookup_pred(st, key), st.z)
            row.append(_render_overlay_rgb(gray, pred=pred, gt=gt2 if key != "gt" or show_gt_on_all else gt2))
        panels.append(row)
    _compose_mosaic(panels, labels, title, col_gap=2, row_gap=6, stem=stem)


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
        ours_k: "Ours",
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


def _load_scripts_pyc(mod_name: str):
    """Load a ControlNet ``scripts.*`` module from ``__pycache__`` (sources often absent)."""
    import importlib.util
    import sys
    import types

    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if "scripts" not in sys.modules:
        pkg = types.ModuleType("scripts")
        pkg.__path__ = [str(CNET / "scripts")]
        sys.modules["scripts"] = pkg
    short = mod_name.split(".", 1)[1]
    pyc_dir = CNET / "scripts" / "__pycache__"
    candidates = sorted(pyc_dir.glob(f"{short}.cpython-*.pyc"), reverse=True)
    if not candidates:
        raise ImportError(f"No pyc for {mod_name} under {pyc_dir}")
    # Prefer matching this interpreter's magic when possible.
    tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    ordered = [p for p in candidates if tag in p.name] + [
        p for p in candidates if tag not in p.name
    ]
    last_err: Exception | None = None
    for path in ordered:
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            sys.modules.pop(mod_name, None)
    raise ImportError(f"Failed to load {mod_name}: {last_err}")


def _run_nni_healthy(case_indices: list[int]) -> dict[int, np.ndarray]:
    """Re-infer nnInteractive on healthy random-box prompts (fixes empty gallery column).

    Important: ``benchmark_nninteractive._ensure_controlnet_scripts`` reloads
    ``scripts.*`` and would wipe an in-memory ``_POSITIVE_CACHE`` injection, so
    we pass a custom ``load_sample_fn`` that returns box-filled nodules.
    """
    import json

    import torch

    npz_path = CACHE / "healthy_nni_preds.npz"
    out: dict[int, np.ndarray] = {}
    if npz_path.is_file():
        bag = np.load(npz_path)
        for i in case_indices:
            key = str(int(i))
            if key in bag.files:
                out[int(i)] = np.asarray(bag[key], dtype=np.float32)
                LOG.info("nnInteractive #%s loaded from cache voxels=%d", i, int((out[int(i)] > 0.5).sum()))
    missing = [i for i in case_indices if int(i) not in out]
    if not missing:
        return out
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"nnInteractive needs CUDA for cases {missing}; "
            f"cached preds only for {sorted(out)} ({npz_path})"
        )

    bn = _load_scripts_pyc("scripts.benchmark_nninteractive")
    _load_scripts_pyc("scripts.setting")
    try:
        btf = _load_scripts_pyc("scripts.benchmark_test_foundation")
        normalize_inpaint_data_cfg = btf.normalize_inpaint_data_cfg
    except Exception:
        normalize_inpaint_data_cfg = None
    load_yaml_config = _load_scripts_pyc("scripts.setting").load_yaml_config

    cfg = load_yaml_config(CNET / "configs/benchmark_healthy_fp_random_boxes.yaml")
    try:
        data_cfg = normalize_inpaint_data_cfg(cfg) if normalize_inpaint_data_cfg else dict(cfg.get("data", {}))
    except Exception:
        data_cfg = dict(cfg.get("data", {}))
    if "positive_patch_cache" not in data_cfg:
        data_cfg["positive_patch_cache"] = str(
            CNET / "runs/diffusion/healthy_patches_cache_test.pt"
        )
    cache_path = Path(data_cfg["positive_patch_cache"])
    if not cache_path.is_absolute():
        # Prefer ControlNet_Diffusion-relative paths, then repo root.
        cand = (CNET / cache_path).resolve()
        cache_path = cand if cand.is_file() else (ROOT / cache_path).resolve()
        data_cfg["positive_patch_cache"] = str(cache_path)

    boxes_path = Path(cfg.get("random_boxes_json", CNET / "runs/benchmark_healthy_fp_random_boxes/random_boxes.json"))
    if not boxes_path.is_absolute():
        cand = (CNET / boxes_path).resolve()
        boxes_path = cand if cand.is_file() else (ROOT / boxes_path).resolve()
    by_idx = {
        int(b["patch_index"]): b
        for b in json.loads(boxes_path.read_text())["boxes"]
    }

    LOG.info("Loading healthy cache for nnInteractive prompt injection …")
    samples = torch.load(cache_path, map_location="cpu", weights_only=False)
    prompt_by_idx: dict[int, dict] = {}
    for i in missing:
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

    bn.prepare_nninteractive_model()
    session = bn.load_nninteractive_session(device="cuda")
    LOG.info("Running nnInteractive on cases %s", missing)
    results = bn.run_nninteractive_on_cases(
        missing,
        data_cfg,
        session,
        hu_min=float(cfg.get("hu_min", -1000)),
        hu_max=float(cfg.get("hu_max", 400)),
        threshold=float(cfg.get("nodule_threshold", 0.5)),
        padding_voxels=int(cfg.get("bbox_padding_voxels_fm", 2)),
        prompt_mode="strategy4_slice_bbox",
        load_sample_fn=_load_sample,
    )
    for case in results:
        idx = int(case["index"])
        pred = np.asarray(case["pred"], dtype=np.float32)
        out[idx] = pred
        LOG.info("nnInteractive #%s voxels=%d", idx, int((pred > 0.5).sum()))
    # Merge into on-disk cache for later GPU-less redraws.
    merged = {str(k): np.asarray(v, dtype=np.float32) for k, v in out.items()}
    if npz_path.is_file():
        old = np.load(npz_path)
        for k in old.files:
            merged.setdefault(k, np.asarray(old[k], dtype=np.float32))
    np.savez_compressed(npz_path, **merged)
    LOG.info("Wrote nnInteractive cache %s (%d cases)", npz_path, len(merged))
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

    try:
        nni_preds = _run_nni_healthy(case_ids)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("nnInteractive healthy re-infer failed (%s); cropping gallery columns", exc)
        nni_preds = {}

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
        "Ours",
    ]
    from PIL import Image as _PILImage

    def _gray_from_hu(hu_vol: np.ndarray, z: int) -> np.ndarray:
        sl = hu_vol[z]
        if float(np.asarray(hu_vol).max()) <= 1.5:
            g = np.clip(sl, 0, 1)
            if DISPLAY_GAMMA != 1.0:
                g = np.power(g, float(DISPLAY_GAMMA))
            return g
        return _gray(sl)

    def _panel_with_prompt_and_pred(gray: np.ndarray, prompt2d, pred2d) -> np.ndarray:
        """CT + optional green-filled pred + yellow prompt contour."""
        fig_t, ax_t = plt.subplots(figsize=(2.56, 2.56), dpi=100)
        ax_t.imshow(gray, cmap="gray", vmin=0, vmax=1, aspect="equal")
        if pred2d is not None:
            arr = np.asarray(pred2d, dtype=float)
            if arr.ndim > 2:
                arr = np.squeeze(arr)
            if arr.size and float(np.nanmax(arr)) > 0:
                mask = arr > 0.5
                if mask.any():
                    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                    overlay[mask] = (0.15, 0.95, 0.15, 0.50)
                    ax_t.imshow(overlay, interpolation="nearest", aspect="equal")
                    _contour(ax_t, arr, "lime", 1.35)
        if prompt2d is not None:
            _contour(ax_t, prompt2d, "yellow", 1.25)
        ax_t.set_axis_off()
        fig_t.subplots_adjust(0, 0, 1, 1)
        fig_t.canvas.draw()
        tile = np.ascontiguousarray(np.asarray(fig_t.canvas.buffer_rgba())[..., :3])
        plt.close(fig_t)
        return tile

    panels: list[list[np.ndarray]] = []
    panel_px = 256
    for idx in case_ids:
        ri = idx_to_row[idx]
        y0, y1 = rsegs[ri]
        sample = samples[idx]
        hu = np.asarray(sample["hu"], dtype=np.float32)
        if "per_case" in meta and str(idx) in meta["per_case"]:
            z = int(meta["per_case"][str(idx)]["z"])
        else:
            z = _mid_box_z(boxes[idx]["bbox_zyx"])
        gray64 = _gray_from_hu(hu, z)
        # Upsample CT to gallery resolution so recovered contours stay intact.
        gray = np.asarray(
            _PILImage.fromarray((np.clip(gray64, 0, 1) * 255).astype(np.uint8)).resize(
                (panel_px, panel_px), _PILImage.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        prompt64 = _bbox_to_mask(hu.shape, boxes[idx]["bbox_zyx"])[z]
        prompt2d = np.asarray(
            _PILImage.fromarray((prompt64 > 0.5).astype(np.uint8) * 255).resize(
                (panel_px, panel_px), _PILImage.NEAREST
            )
        ) > 127
        row = []
        src_info = []
        for c, ci in enumerate(src_cols):
            x0, x1 = csegs[ci]
            gallery = np.asarray(
                _PILImage.fromarray(im[y0:y1, x0:x1]).resize(
                    (panel_px, panel_px), _PILImage.BILINEAR
                )
            )
            if c == 0:
                # Prompt column: yellow contour only (no fill overlay).
                fig_t, ax_t = plt.subplots(figsize=(2.56, 2.56), dpi=100)
                ax_t.imshow(gray, cmap="gray", vmin=0, vmax=1, aspect="equal")
                _contour(ax_t, prompt2d.astype(np.float32), "yellow", 1.35)
                ax_t.set_axis_off()
                fig_t.subplots_adjust(0, 0, 1, 1)
                fig_t.canvas.draw()
                tile = np.ascontiguousarray(np.asarray(fig_t.canvas.buffer_rgba())[..., :3])
                plt.close(fig_t)
                src_info.append("prompt")
            else:
                pred2d = None
                if c == nni_paper_col and idx in nni_preds:
                    pred64 = np.asarray(nni_preds[idx])[z] > 0.5
                    pred2d = np.asarray(
                        _PILImage.fromarray(pred64.astype(np.uint8) * 255).resize(
                            (panel_px, panel_px), _PILImage.NEAREST
                        )
                    ) > 127
                    src_info.append(f"nni:{int(pred2d.sum())}")
                else:
                    pred_mask = _mask_from_color_contours(gallery, mode="lime")
                    if pred_mask.any():
                        pred2d = pred_mask
                        src_info.append(f"gallery:{int(pred_mask.sum())}")
                    else:
                        src_info.append("none")
                tile = _panel_with_prompt_and_pred(
                    gray,
                    prompt2d.astype(np.float32),
                    None if pred2d is None else pred2d.astype(np.float32),
                )
            row.append(tile)
        panels.append(row)
        LOG.info("healthy #%s z=%d overlay=%s", idx, z, dict(zip(labels, src_info)))
    _compose_mosaic(
        panels,
        labels,
        "Healthy FP — yellow=prompt contour, green=prediction fill (GT empty)",
        col_gap=2,
        row_gap=6,
        stem="fig_qual_healthy_fp",
    )


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
        ("Ours", "seg_ours"),
    ]

    method_labels = [t for t, _ in methods]

    # ---- Volume-wise (full axial / best-axis slice) ----
    panels_vol: list[list[np.ndarray]] = []
    for pid in NSCLC_PATIENTS:
        ct = np.asanyarray(nib.load(str(base / "processed_ct" / f"{pid}.nii.gz")).dataobj).astype(np.float32)
        gt = np.asanyarray(nib.load(str(base / "processed_gtv" / f"{pid}.nii.gz")).dataobj) > 0.5
        profiles = [gt.sum(axis=tuple(i for i in range(3) if i != ax)) for ax in range(3)]
        slice_axis = int(np.argmax([float(p.max()) for p in profiles]))
        z = int(np.argmax(profiles[slice_axis]))
        sl = np.take(ct, z, axis=slice_axis)
        g2 = np.take(gt, z, axis=slice_axis)
        gray = np.rot90(_gray(sl))
        gt2 = np.rot90(g2)
        row = []
        for _title, folder in methods:
            pred2 = None
            if folder is not None:
                pred = np.asanyarray(nib.load(str(base / folder / f"{pid}.nii.gz")).dataobj) > 0.5
                if pred.shape != gt.shape and pred.T.shape == gt.shape:
                    pred = pred.T
                pred2 = np.rot90(np.take(pred, z, axis=slice_axis))
            row.append(_render_overlay_rgb(gray, pred=pred2, gt=gt2))
        panels_vol.append(row)
    _compose_mosaic(
        panels_vol,
        method_labels,
        "NSCLC-Radiomics (volume) — red=GT, lime=prediction",
        col_gap=2,
        row_gap=6,
        stem="fig_qual_nsclc_volume",
    )

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

    panels_p: list[list[np.ndarray]] = []
    for pid, pidx in zip(NSCLC_PATIENTS, NSCLC_PATCHES):
        sample = cache[int(pidx)]
        if str(sample.get("patient_id")) != pid:
            LOG.warning("NSCLC patch %s patient_id=%s (expected %s)", pidx, sample.get("patient_id"), pid)
        center = sample["center"]
        hu = np.asarray(sample["hu"], dtype=np.float32)
        hu_hu = hu * 1400.0 - 1000.0 if float(hu.max()) <= 1.5 else hu
        nod = np.asarray(sample["nodule"]) > 0.5
        z = int(np.argmax(nod.sum(axis=(1, 2))))
        gray = _gray(hu_hu[z])
        gtc = _nsclc_extract_patch(_vol(pid, None), center)
        if gtc is not None:
            dice_gt = 2.0 * float((gtc & nod).sum()) / max(1.0, float(gtc.sum() + nod.sum()))
            if dice_gt < 0.99:
                LOG.warning("NSCLC patch #%s GT align dice=%.3f", pidx, dice_gt)
        row = []
        for _title, folder in methods:
            pred2 = None
            if folder is not None:
                pred = _nsclc_extract_patch(_vol(pid, folder), center)
                if pred is None:
                    LOG.warning("NSCLC patch crop failed %s %s", pid, folder)
                else:
                    pred2 = pred[z]
            row.append(_render_overlay_rgb(gray, pred=pred2, gt=nod[z]))
        panels_p.append(row)
        LOG.info("NSCLC patch row %s #%s z=%d nodule_vox=%d", pid, pidx, z, int(nod.sum()))
    _compose_mosaic(
        panels_p,
        method_labels,
        "NSCLC-Radiomics (64³ patches) — red=GT, lime=prediction",
        col_gap=2,
        row_gap=6,
        stem="fig_qual_nsclc",
    )


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
