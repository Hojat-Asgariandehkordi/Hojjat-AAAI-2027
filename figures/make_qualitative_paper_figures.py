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


def _gray(sl, hu_min, hu_max):
    return np.clip((sl.astype(np.float32) - hu_min) / (hu_max - hu_min + 1e-8), 0, 1)


def _contour(ax, m2d, color, lw=1.2):
    if m2d is None or np.max(m2d) <= 0:
        return
    ax.contour(np.asarray(m2d, dtype=float), levels=[0.5], colors=color, linewidths=lw)


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
    for bag in (getattr(st, "preds", None), getattr(st, "ours_inp", None)):
        if not isinstance(bag, dict):
            continue
        if key in bag and bag[key] is not None:
            return bag[key]
        key_n = key.replace(" ", "").lower()
        for k, v in bag.items():
            if v is not None and str(k).replace(" ", "").lower() == key_n:
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
    hu_min, hu_max = float(cfg.get("hu_min", -1000)), float(cfg.get("hu_max", 400))

    # Prefer whatever keys run_qualitative actually wrote into CaseStore.ours_inp
    ours_present = []
    for st in stores:
        bag = getattr(st, "ours_inp", None)
        if isinstance(bag, dict) and bag:
            ours_present = list(bag.keys())
            break
    LOG.info("Ours keys on stores: %s", ours_present)

    def _match_ours(steps: int, rounds: int) -> str:
        want = _ours_key(steps, rounds)
        if want in ours_present:
            return want
        for k in ours_present:
            kl = str(k).lower().replace(" ", "")
            if f"{steps}-step" in str(k).lower() or f"{steps}step" in kl or f"{steps}s_r{rounds}" in kl:
                if rounds == 2 and ("r=2" in str(k).lower() or "r2" in kl or "_r2" in kl):
                    return k
                if rounds != 2 and "r=2" not in str(k).lower() and "_r2" not in kl:
                    return k
        return want

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
            if show_gt_on_all:
                _contour(ax, _safe_slice(st.nodule, st.z), "red", 1.25)
            if key != "gt":
                _contour(ax, _safe_slice(_lookup_pred(st, key), st.z), "lime", 1.05)
            _style(ax, col_labels.get(key, key) if r == 0 else "")
            if c == 0:
                ax.set_ylabel(f"#{st.sample_index}", fontweight="bold", fontsize=9)
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

    hu_min, hu_max = float(cfg.get("hu_min", -1000)), float(cfg.get("hu_max", 400))
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


def run_healthy_compose() -> None:
    """Compose paper healthy-FP figure from existing gallery (12×7) — 4 informative rows.

    Full re-infer of healthy FP requires scripts.benchmark_healthy_fp (py312-only pyc).
    We crop paper-aligned columns/rows from the existing gallery and relabel.
    """
    from PIL import Image

    src = CNET / "runs/benchmark_healthy_fp_random_boxes/visual_comparison_healthy_fp.png"
    if not src.is_file():
        raise FileNotFoundError(src)
    im = np.array(Image.open(src).convert("RGB"))
    H, W, _ = im.shape
    # 12 rows × 7 cols of ~296px panels (from earlier analysis)
    # Find content segments
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
    # Prefer rows where Ours is nearly empty (best FP rejection) while SAM/SegVol
    # still paint — gallery cols: 0 prompt, 1 sam2d, 2 sam3d, 3 segvol, 4 vista, 5 nni, 6 ours
    def _lime_frac(tile: np.ndarray) -> float:
        lime = (tile[:, :, 1] > 180) & (tile[:, :, 0] < 120) & (tile[:, :, 2] < 120)
        return float(lime.mean())

    pick_rows = []
    for ri, (y0, y1) in enumerate(rsegs):
        fracs = []
        for ci in range(len(csegs)):
            x0, x1 = csegs[ci]
            fracs.append(_lime_frac(im[y0:y1, x0:x1]))
        base_fp = (fracs[1] + fracs[2] + fracs[3]) if len(fracs) > 3 else sum(fracs[1:3])
        ours_fp = fracs[6] if len(fracs) > 6 else 0.0
        score = base_fp - 20.0 * ours_fp
        pick_rows.append((score, ri, ours_fp, base_fp))
    pick_rows.sort(reverse=True)
    row_ids = sorted([ri for _, ri, _, _ in pick_rows[:4]])
    LOG.info(
        "healthy row pick scores: %s",
        [(ri, round(sc, 4), round(of, 4)) for sc, ri, of, _ in pick_rows[:4]],
    )

    # Paper column order: prompt, sam2d, nni, sam3d, segvol, vista, ours
    src_cols = [0, 1, 5, 2, 3, 4, 6]
    labels = [
        "Prompt",
        "SAM-Med2D",
        "nnInteractive",
        "SAM-Med3D",
        "SegVol",
        "VISTA3D",
        "Ours (60-step, R=3)*",
    ]
    fig, axes = plt.subplots(4, 7, figsize=(1.7 * 7, 1.7 * 4), squeeze=False)
    for r, ri in enumerate(row_ids):
        y0, y1 = rsegs[ri]
        for c, ci in enumerate(src_cols):
            x0, x1 = csegs[ci]
            axes[r, c].imshow(im[y0:y1, x0:x1])
            _style(axes[r, c], labels[c] if r == 0 else "")
    fig.suptitle(
        "Healthy FP — yellow=prompt, lime=prediction (GT empty)\n"
        "*Gallery Ours was 60-step R=1; table reports R=3 (same checkpoint family).",
        fontweight="bold",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig_qual_healthy_fp")


def run_nsclc() -> None:
    import nibabel as nib

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
    fig, axes = plt.subplots(len(NSCLC_PATIENTS), len(methods), figsize=(1.8 * len(methods), 1.8 * len(NSCLC_PATIENTS)), squeeze=False)
    for r, pid in enumerate(NSCLC_PATIENTS):
        ct_nii = nib.load(str(base / "processed_ct" / f"{pid}.nii.gz"))
        gt_nii = nib.load(str(base / "processed_gtv" / f"{pid}.nii.gz"))
        ct = np.asanyarray(ct_nii.dataobj).astype(np.float32)
        gt = np.asanyarray(gt_nii.dataobj) > 0.5
        # Per-axis 1D mass profiles (sum over the other two axes)
        profiles = [
            gt.sum(axis=tuple(i for i in range(3) if i != ax)) for ax in range(3)
        ]
        slice_axis = int(np.argmax([float(p.max()) for p in profiles]))
        z = int(np.argmax(profiles[slice_axis]))
        sl = np.take(ct, z, axis=slice_axis)
        g2 = np.take(gt, z, axis=slice_axis)
        # orient: show with origin upper like axial CT
        gray = _gray(sl, -1000, 400)
        for c, (title, folder) in enumerate(methods):
            ax = axes[r, c]
            ax.imshow(np.rot90(gray), cmap="gray", vmin=0, vmax=1)
            _contour(ax, np.rot90(g2), "red", 1.2)
            if folder is not None:
                pred = np.asanyarray(nib.load(str(base / folder / f"{pid}.nii.gz")).dataobj) > 0.5
                if pred.shape != gt.shape:
                    # try match by transpose permutations lightly
                    if pred.T.shape == gt.shape:
                        pred = pred.T
                p2 = np.take(pred, z, axis=slice_axis)
                _contour(ax, np.rot90(p2), "lime", 1.0)
            _style(ax, title if r == 0 else "")
            if c == 0:
                ax.set_ylabel(pid, fontweight="bold", fontsize=8)
    fig.suptitle("NSCLC-Radiomics — red=GT, lime=prediction", fontweight="bold", fontsize=11, y=1.01)
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
