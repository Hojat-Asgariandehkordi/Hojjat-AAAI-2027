#!/usr/bin/env python3
"""Export PNG assets for bakeoff/method_overview/fig_method_overview(2).tikz.tex.

Uses LIDC qualitative CaseStore (paper_qual_cache/lidc_stores.pkl), default #1370.
Run with the nninteractive env (numpy 2.x pickle compatible):

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
from PIL import Image

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR))

from make_method_overview import (  # noqa: E402
    CACHE,
    _bbox_from_mask,
    _draw_box,
    _gray,
    _load_panels,
    _overlay_contour,
    _rgb_gray,
    _to_np,
)

OUT = FIG_DIR / "bakeoff" / "method_overview" / "assets"


def _save(name: str, rgb: np.ndarray, size: int = 256) -> None:
    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).resize((size, size), Image.NEAREST).save(OUT / name)
    print(f"Wrote {OUT / name}")


def main(case_index: int = 1370) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels_train, _, case_id = _load_panels(case_index)
    train = {t.replace("\n", " "): img for t, img in panels_train}

    stores = pickle.loads((CACHE / "lidc_stores.pkl").read_bytes())
    st = next((s for s in stores if int(s["sample_index"]) == case_index), stores[0])
    z = int(st["z"])
    hu = _to_np(st["gt_hu"])
    nod = _to_np(st["nodule"]) > 0.5
    pred = _to_np(st["preds"]["ours:40s_r2"]) > 0.5
    inp = np.squeeze(_to_np(st["ours_inp"]["40s_r2"]))
    g, n2, p2, i2 = _gray(hu[z]), nod[z], pred[z], _gray(inp[z])
    bbox = _bbox_from_mask(n2, pad=2)
    res_n = np.abs(g - i2)
    res_n = res_n / (res_n.max() + 1e-6)
    thr = res_n > 0.28
    heat = plt.cm.magma(res_n)[..., :3]

    hole = np.zeros_like(n2, dtype=bool)
    if bbox is not None:
        y0, x0, y1, x1 = bbox
        hole[y0 : y1 + 1, x0 : x1 + 1] = True
    hole_img = _rgb_gray(g)
    hole_img[hole] = (0.05, 0.05, 0.05)
    hole_img = _draw_box(hole_img, bbox)
    resid = 0.45 * _rgb_gray(g) + 0.55 * heat
    final = _overlay_contour(_rgb_gray(g), p2, (0.2, 0.95, 0.35), width=1)
    final = _overlay_contour(final, n2, (0.95, 0.15, 0.12), width=1)
    refined = _overlay_contour(_rgb_gray(g), thr | p2, (0.2, 0.95, 0.35), width=1)

    _save("healthy_patch.png", _rgb_gray(i2))
    _save("noise_patch.png", train["Random Noise"])
    _save("rf_training.png", train["Rectified Flow Training"])
    _save("velocity_loss.png", train["Velocity Loss"])
    _save("learned_prior.png", train["Learned Healthy Prior"])
    _save("test_ct.png", _rgb_gray(g))
    _save("box_prompt.png", _draw_box(_rgb_gray(g), bbox))
    _save("initial_hole.png", hole_img)
    _save("healthy_reconstruction.png", _rgb_gray(i2))
    _save("residual.png", resid)
    _save("threshold.png", _overlay_contour(_rgb_gray(g * 0.55), thr, (0.2, 0.95, 0.35), width=1))
    _save("final_seg.png", final)
    _save("iter1.png", hole_img, size=320)
    _save("residual1.png", resid, size=320)
    _save("iter2.png", refined, size=320)
    _save("final_zoom.png", final, size=320)

    (OUT / "README.txt").write_text(
        f"TikZ assets from LIDC case #{case_id} (z={z}).\n"
        "Used by fig_method_overview(2).tikz.tex\n"
    )
    print(f"Done — case #{case_id} → {OUT}")


if __name__ == "__main__":
    main()
