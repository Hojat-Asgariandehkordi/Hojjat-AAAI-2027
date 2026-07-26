#!/usr/bin/env python3
"""Export PNG assets for method-overview figures (TikZ / matplotlib / draw.io).

Each CT panel is a **3-slice stack** (z−1, z, z+1) to emphasize 3D 64³ patches.
Uses LIDC qualitative CaseStore (paper_qual_cache/lidc_stores.pkl), default #1370.

  /home/morteza/.conda/envs/nninteractive/bin/python \\
      figures/export_method_overview_tikz_assets.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR))

from make_method_overview import (  # noqa: E402
    CACHE,
    _bbox_from_mask,
    _draw_box,
    _gray,
    _overlay_contour,
    _rgb_gray,
    _to_np,
)

OUT = FIG_DIR / "bakeoff" / "method_overview" / "assets"
N_STACK = 3  # consecutive slices centered on the nodule plane


def _save(name: str, rgb: np.ndarray, size: int | None = 256) -> None:
    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if size is not None:
        # keep aspect for stacks (wider than tall)
        h, w = arr.shape[:2]
        if w >= h:
            im = im.resize((size, max(1, int(size * h / w))), Image.NEAREST)
        else:
            im = im.resize((max(1, int(size * w / h)), size), Image.NEAREST)
    im.save(OUT / name)
    print(f"Wrote {OUT / name}  shape={arr.shape[:2]}")


def _slice_ids(z: int, n: int, zmax: int) -> list[int]:
    half = n // 2
    ids = [z + k - half for k in range(n)]
    # clamp into volume
    ids = [int(np.clip(i, 0, zmax - 1)) for i in ids]
    # if clamping collapsed range, expand uniquely around z
    if len(set(ids)) < n:
        ids = sorted({int(np.clip(z + d, 0, zmax - 1)) for d in range(-(n - 1), n)})[:n]
        while len(ids) < n:
            ids.append(ids[-1])
    return ids


def _compose_stack(slices_rgb: list[np.ndarray], *, offset_frac: float = 0.20) -> np.ndarray:
    """Fan 3 RGB slices into one image (back → front) to suggest a 3D volume."""
    assert len(slices_rgb) >= 1
    h, w = slices_rgb[0].shape[:2]
    n = len(slices_rgb)
    ox = max(2, int(round(offset_frac * w)))
    oy = max(2, int(round(offset_frac * h)))
    canvas_h = h + oy * (n - 1) + 2
    canvas_w = w + ox * (n - 1) + 2
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)
    # draw back-to-front
    for i, sl in enumerate(slices_rgb):
        # i=0 is farthest (top-left), last is front
        x0 = i * ox
        y0 = i * oy
        canvas[y0 : y0 + h, x0 : x0 + w] = np.clip(sl, 0, 1)
        # thin border so layers separate
        canvas[y0 : y0 + h, x0] = 0.15
        canvas[y0 : y0 + h, x0 + w - 1] = 0.15
        canvas[y0, x0 : x0 + w] = 0.15
        canvas[y0 + h - 1, x0 : x0 + w] = 0.15
    return canvas


def _render_panel_at_z(
    hu_g: np.ndarray,
    inp_g: np.ndarray,
    nod: np.ndarray,
    pred: np.ndarray,
    zi: int,
    kind: str,
) -> np.ndarray:
    """Render one axial slice panel for a named kind."""
    g = hu_g[zi]
    i2 = inp_g[zi]
    n2 = nod[zi]
    p2 = pred[zi]
    bbox = _bbox_from_mask(n2, pad=2)

    if kind == "healthy":
        return _rgb_gray(i2)
    if kind == "noise":
        rng = np.random.default_rng(zi + 7)
        noise = np.clip(rng.normal(0.45, 0.22, size=g.shape).astype(np.float32), 0, 1)
        return _rgb_gray(noise)
    if kind == "prior":
        try:
            from scipy.ndimage import gaussian_filter

            return _rgb_gray(gaussian_filter(i2, sigma=0.9))
        except Exception:
            return _rgb_gray(i2)
    if kind == "ct":
        return _rgb_gray(g)
    if kind == "box":
        return _draw_box(_rgb_gray(g), bbox)
    if kind == "hole":
        hole = np.zeros_like(n2, dtype=bool)
        if bbox is not None:
            y0, x0, y1, x1 = bbox
            hole[y0 : y1 + 1, x0 : x1 + 1] = True
        img = _rgb_gray(g)
        img[hole] = (0.05, 0.05, 0.05)
        return _draw_box(img, bbox)
    if kind == "recon":
        return _rgb_gray(i2)
    if kind == "residual":
        res = np.abs(g - i2)
        res_n = res / (res.max() + 1e-6)
        heat = plt.cm.magma(res_n)[..., :3]
        return 0.45 * _rgb_gray(g) + 0.55 * heat
    if kind == "threshold":
        res = np.abs(g - i2)
        res_n = res / (res.max() + 1e-6)
        thr = res_n > 0.28
        return _overlay_contour(_rgb_gray(g * 0.55), thr, (0.2, 0.95, 0.35), width=1)
    if kind == "refined":
        res = np.abs(g - i2)
        res_n = res / (res.max() + 1e-6)
        thr = res_n > 0.28
        return _overlay_contour(_rgb_gray(g), thr | p2, (0.2, 0.95, 0.35), width=1)
    if kind == "final":
        out = _overlay_contour(_rgb_gray(g), p2, (0.2, 0.95, 0.35), width=1)
        return _overlay_contour(out, n2, (0.95, 0.15, 0.12), width=1)
    raise ValueError(kind)


def _stack_for(kind: str, z_ids: list[int], hu_g, inp_g, nod, pred) -> np.ndarray:
    slices = [_render_panel_at_z(hu_g, inp_g, nod, pred, zi, kind) for zi in z_ids]
    return _compose_stack(slices)


def _annotate_z_strip(rgb: np.ndarray, z_ids: list[int]) -> np.ndarray:
    """Optional tiny footer labels baked into a copy (for standalone assets)."""
    # keep clean for paper; z labels are in the figure caption / panel caption instead
    return rgb


def main(case_index: int = 1370) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stores = pickle.loads((CACHE / "lidc_stores.pkl").read_bytes())
    st = next((s for s in stores if int(s["sample_index"]) == case_index), stores[0])
    z = int(st["z"])
    hu = _to_np(st["gt_hu"])
    nod = (_to_np(st["nodule"]) > 0.5)
    pred = (_to_np(st["preds"]["ours:40s_r2"]) > 0.5)
    inp = np.squeeze(_to_np(st["ours_inp"]["40s_r2"]))

    # gray volumes once (shared windowing per-slice still via _gray)
    hu_g = np.stack([_gray(hu[zi]) for zi in range(hu.shape[0])], axis=0)
    inp_g = np.stack([_gray(inp[zi]) for zi in range(inp.shape[0])], axis=0)
    z_ids = _slice_ids(z, N_STACK, hu.shape[0])
    print(f"Case #{int(st['sample_index'])}  center z={z}  stack z={z_ids}")

    mapping = {
        "healthy_patch.png": "healthy",
        "noise_patch.png": "noise",
        "learned_prior.png": "prior",
        "test_ct.png": "ct",
        "box_prompt.png": "box",
        "initial_hole.png": "hole",
        "healthy_reconstruction.png": "recon",
        "residual.png": "residual",
        "threshold.png": "threshold",
        "final_seg.png": "final",
        "iter1.png": "hole",
        "residual1.png": "residual",
        "iter2.png": "refined",
        "final_zoom.png": "final",
    }
    # keep a few non-stack decorative panels for RF/loss (still 3D-looking noise/mix)
    for name, kind in mapping.items():
        stack = _annotate_z_strip(_stack_for(kind, z_ids, hu_g, inp_g, nod, pred), z_ids)
        _save(name, stack, size=280)

    # RF training / velocity: stack of mixed noise+prior slices
    rf_slices = []
    vel_slices = []
    rng = np.random.default_rng(0)
    for zi in z_ids:
        healthy = inp_g[zi]
        noise = np.clip(rng.normal(0.45, 0.22, size=healthy.shape).astype(np.float32), 0, 1)
        try:
            from scipy.ndimage import gaussian_filter

            rf = gaussian_filter(0.55 * noise + 0.45 * healthy, sigma=0.6)
        except Exception:
            rf = 0.55 * noise + 0.45 * healthy
        rf_slices.append(_rgb_gray(rf))
        vel_slices.append(_rgb_gray(np.clip(np.abs(noise - healthy) * 1.4, 0, 1)))
    _save("rf_training.png", _compose_stack(rf_slices), size=280)
    _save("velocity_loss.png", _compose_stack(vel_slices), size=280)

    # also export individual slices for optional use
    slice_dir = OUT / "slices"
    slice_dir.mkdir(exist_ok=True)
    for kind in ("ct", "box", "hole", "recon", "residual", "final", "healthy"):
        for zi in z_ids:
            rgb = _render_panel_at_z(hu_g, inp_g, nod, pred, zi, kind)
            arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(arr).resize((128, 128), Image.NEAREST).save(slice_dir / f"{kind}_z{zi}.png")

    (OUT / "README.txt").write_text(
        f"TikZ/matplotlib/draw.io assets from LIDC case #{int(st['sample_index'])}.\n"
        f"Center slice z={z}; stack shows consecutive planes {z_ids} (3D 64³).\n"
        "Regenerate: nninteractive python figures/export_method_overview_tikz_assets.py\n"
    )
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
