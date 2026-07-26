#!/usr/bin/env python3
"""Camera-ready method flowchart with CT / mask sample thumbnails.

Uses one LIDC qualitative CaseStore (paper_qual_cache/lidc_stores.pkl) so the
diagram shows real patches: CT+box, hole, healthy reconstruction, residual,
threshold, and final mask. Training-side panels are derived from the same CT
(healthy-looking parenchyma + synthetic noise).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FIG_DIR = Path(__file__).resolve().parent
CACHE = FIG_DIR / "paper_qual_cache"
STEM = "fig_method_overview"

C_TRAIN = "#0B3D5C"
C_INFER = "#047857"
C_EDGE = "#1F2937"
C_ARROW = "#4B5563"
C_LOOP = "#B45309"
C_PANEL = "#F3F4F6"
C_OUT = "#065F46"


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _gray(sl: np.ndarray) -> np.ndarray:
    x = _to_np(sl).astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo + 1e-6:
        lo, hi = float(x.min()), float(x.max() + 1e-6)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _rgb_gray(g: np.ndarray) -> np.ndarray:
    g = np.clip(g, 0, 1)
    return np.stack([g, g, g], axis=-1)


def _overlay_contour(rgb: np.ndarray, mask: np.ndarray, color, width: int = 1) -> np.ndarray:
    """Draw a thin contour of a binary mask onto an RGB image."""
    from scipy import ndimage

    m = _to_np(mask).astype(bool)
    if not m.any():
        return rgb
    # boundary = mask XOR eroded
    er = ndimage.binary_erosion(m, iterations=max(1, width))
    edge = m & ~er
    if width > 1:
        edge = ndimage.binary_dilation(edge, iterations=width - 1)
    out = rgb.copy()
    out[edge] = np.asarray(color, dtype=np.float32)
    return out


def _bbox_from_mask(mask: np.ndarray, pad: int = 2):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()) - pad, int(ys.max()) + pad
    x0, x1 = int(xs.min()) - pad, int(xs.max()) + pad
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(mask.shape[0] - 1, y1), min(mask.shape[1] - 1, x1)
    return y0, x0, y1, x1


def _draw_box(rgb: np.ndarray, bbox, color=(1.0, 0.92, 0.1), t: int = 1) -> np.ndarray:
    if bbox is None:
        return rgb
    y0, x0, y1, x1 = bbox
    out = rgb.copy()
    out[y0 : y0 + t, x0 : x1 + 1] = color
    out[y1 : y1 + t, x0 : x1 + 1] = color
    out[y0 : y1 + 1, x0 : x0 + t] = color
    out[y0 : y1 + 1, x1 : x1 + t] = color
    return out


def _load_panels(case_index: int = 1370):
    pkl = CACHE / "lidc_stores.pkl"
    stores = pickle.loads(pkl.read_bytes())
    st = None
    for s in stores:
        if int(s["sample_index"]) == case_index:
            st = s
            break
    if st is None:
        st = stores[0]
    z = int(st["z"])
    hu = _to_np(st["gt_hu"])
    nod = _to_np(st["nodule"]) > 0.5
    pred = _to_np(st["preds"]["ours:40s_r2"]) > 0.5
    # Healthy reconstruction (inpainted HU volume); may be (1,Z,Y,X)
    inp = np.squeeze(_to_np(st["ours_inp"]["40s_r2"]))

    g = _gray(hu[z])
    n2 = nod[z]
    p2 = pred[z]
    i2 = _gray(inp[z])
    bbox = _bbox_from_mask(n2, pad=2)

    # Hole from box (strategy-4 style): fill bbox region
    hole = np.zeros_like(n2, dtype=bool)
    if bbox is not None:
        y0, x0, y1, x1 = bbox
        hole[y0 : y1 + 1, x0 : x1 + 1] = True
    else:
        hole = n2

    # Residual (display on CT intensity scale)
    res = np.abs(g - i2)
    res_n = res / (res.max() + 1e-6)
    thr = res_n > 0.28

    # Training panels — use the healthy reconstruction as the "healthy CT"
    # exemplar (avoids a dark hole where the nodule was).
    healthy = i2.copy()
    rng = np.random.default_rng(0)
    noise = rng.normal(0.45, 0.22, size=g.shape).astype(np.float32)
    noise = np.clip(noise, 0, 1)

    try:
        from scipy.ndimage import gaussian_filter

        prior = gaussian_filter(healthy, sigma=0.9)
        rf_mix = gaussian_filter(0.55 * noise + 0.45 * healthy, sigma=0.6)
    except Exception:
        prior = healthy
        rf_mix = 0.55 * noise + 0.45 * healthy

    panels_train = [
        ("Healthy CT Patch", _rgb_gray(healthy)),
        ("Random Noise", _rgb_gray(noise)),
        ("Rectified Flow\nTraining", _rgb_gray(rf_mix)),
        ("Velocity Loss", _rgb_gray(np.clip(np.abs(noise - healthy) * 1.4, 0, 1))),
        ("Learned Healthy\nPrior", _rgb_gray(prior)),
    ]

    ct_box = _draw_box(_rgb_gray(g), bbox)
    hole_img = _rgb_gray(g)
    hole_img[hole] = (0.05, 0.05, 0.05)
    hole_img = _draw_box(hole_img, bbox, color=(1.0, 0.92, 0.1), t=1)

    rf_img = hole_img.copy()
    # faint lime on hole edge to hint inpainting region
    rf_img = _overlay_contour(rf_img, hole, (0.2, 0.95, 0.35), width=1)

    recon = _rgb_gray(i2)
    # residual as hot overlay on gray
    resid_rgb = _rgb_gray(g)
    heat = plt.cm.magma(res_n)[..., :3]
    resid_rgb = 0.45 * resid_rgb + 0.55 * heat

    thr_rgb = _rgb_gray(g * 0.55)
    thr_rgb = _overlay_contour(thr_rgb, thr, (0.2, 0.95, 0.35), width=1)

    refined = _rgb_gray(g)
    refined = _overlay_contour(refined, thr | p2, (0.2, 0.95, 0.35), width=1)

    final = _rgb_gray(g)
    final = _overlay_contour(final, p2, (0.2, 0.95, 0.35), width=1)
    final = _overlay_contour(final, n2, (0.95, 0.15, 0.12), width=1)

    panels_infer = [
        ("CT Patch +\nBounding Box", ct_box),
        ("Create Hole", hole_img),
        ("Mask-guided RF\nInpainting", rf_img),
        ("Healthy\nReconstruction", recon),
        ("Residual", resid_rgb),
        ("Threshold", thr_rgb),
        ("Refined Hole", refined),
        ("Final\nSegmentation", final),
    ]
    return panels_train, panels_infer, int(st["sample_index"])


def _card(ax, x, y, w, h, title, img, *, title_color=C_EDGE, accent=None):
    # outer card
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            linewidth=1.1,
            edgecolor=accent or C_EDGE,
            facecolor="white",
            zorder=2,
        )
    )
    # image area
    img_pad = 0.06
    img_h = h * 0.62
    img_y = y + h * 0.30
    ax.imshow(
        img,
        extent=(x + img_pad, x + w - img_pad, img_y, img_y + img_h),
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h * 0.14,
        title,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=title_color,
        zorder=4,
        linespacing=1.15,
    )


def _arrow(ax, x0, y0, x1, y1, *, color=C_ARROW):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=11),
        zorder=5,
    )


def draw() -> None:
    panels_train, panels_infer, case_id = _load_panels(1370)

    fig_w, fig_h = 13.6, 7.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- TRAINING row ----
    ax.add_patch(Rectangle((0.25, 5.85), fig_w - 0.5, 0.45, facecolor=C_TRAIN, edgecolor=C_TRAIN, zorder=1))
    ax.text(
        fig_w / 2,
        6.07,
        "TRAINING  (offline)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
        zorder=2,
    )

    n_t = len(panels_train)
    card_w, card_h = 2.15, 2.35
    gap = 0.28
    total_w = n_t * card_w + (n_t - 1) * gap
    x0 = (fig_w - total_w) / 2
    y_t = 3.30
    centers_t = []
    for i, (title, img) in enumerate(panels_train):
        x = x0 + i * (card_w + gap)
        accent = C_TRAIN if i == n_t - 1 else C_EDGE
        tc = C_TRAIN if i == n_t - 1 else C_EDGE
        _card(ax, x, y_t, card_w, card_h, title, img, title_color=tc, accent=accent)
        centers_t.append(x + card_w / 2)
        if i < n_t - 1:
            _arrow(ax, x + card_w + 0.02, y_t + card_h * 0.55, x + card_w + gap - 0.02, y_t + card_h * 0.55)

    # ---- INFERENCE row ----
    ax.add_patch(Rectangle((0.25, 2.55), fig_w - 0.5, 0.45, facecolor=C_INFER, edgecolor=C_INFER, zorder=1))
    ax.text(
        fig_w / 2,
        2.77,
        "INFERENCE  (online)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
        zorder=2,
    )

    n_i = len(panels_infer)
    card_w_i, card_h_i = 1.48, 2.05
    gap_i = 0.14
    total_i = n_i * card_w_i + (n_i - 1) * gap_i
    x0_i = (fig_w - total_i) / 2
    y_i = 0.28
    centers_i = []
    for i, (title, img) in enumerate(panels_infer):
        x = x0_i + i * (card_w_i + gap_i)
        accent = C_OUT if i == n_i - 1 else (C_INFER if i == 2 else C_EDGE)
        tc = C_OUT if i == n_i - 1 else C_EDGE
        _card(ax, x, y_i, card_w_i, card_h_i, title, img, title_color=tc, accent=accent)
        centers_i.append((x, x + card_w_i / 2, x + card_w_i))
        if i < n_i - 1:
            _arrow(
                ax,
                x + card_w_i + 0.01,
                y_i + card_h_i * 0.58,
                x + card_w_i + gap_i - 0.01,
                y_i + card_h_i * 0.58,
                color=C_INFER if i >= n_i - 2 else C_ARROW,
            )

    # iterate R rounds: from Refined Hole (index 6) back to Mask-guided RF (index 2)
    x_rf0, cx_rf, x_rf1 = centers_i[2]
    x_rh0, cx_rh, x_rh1 = centers_i[6]
    loop_y = y_i + card_h_i + 0.12
    ax.plot([cx_rh, cx_rh], [y_i + card_h_i, loop_y], color=C_LOOP, lw=1.35, zorder=5)
    ax.plot([cx_rh, cx_rf], [loop_y, loop_y], color=C_LOOP, lw=1.35, zorder=5)
    ax.annotate(
        "",
        xy=(cx_rf, y_i + card_h_i + 0.01),
        xytext=(cx_rf, loop_y),
        arrowprops=dict(arrowstyle="-|>", color=C_LOOP, lw=1.35, mutation_scale=10),
        zorder=5,
    )
    ax.text(
        (cx_rf + cx_rh) / 2,
        loop_y + 0.10,
        r"iterate $R$ rounds",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=C_LOOP,
        zorder=6,
    )

    # use prior: training last card → inference RF card
    ax.annotate(
        "",
        xy=(cx_rf, y_i + card_h_i + 0.22),
        xytext=(centers_t[-1], y_t - 0.02),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_TRAIN,
            lw=1.2,
            ls=(0, (4, 2)),
            mutation_scale=10,
            connectionstyle="arc3,rad=0.12",
        ),
        zorder=4,
    )
    ax.text(
        (centers_t[-1] + cx_rf) / 2 + 0.6,
        (y_t + y_i + card_h_i) / 2 + 0.15,
        "use prior",
        ha="center",
        va="center",
        fontsize=8,
        color=C_TRAIN,
        style="italic",
        zorder=6,
    )

    ax.text(
        0.35,
        0.08,
        f"Example LIDC patch #{case_id}  ·  red=GT, lime=pred / residual FG",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#6B7280",
        zorder=6,
    )

    fig.tight_layout(pad=0.25)
    for ext in ("pdf", "svg", "png"):
        out = FIG_DIR / f"{STEM}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    draw()
