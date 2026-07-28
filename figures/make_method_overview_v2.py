#!/usr/bin/env python3
"""Camera-ready method overview v2 — strict grid, no overlapping text/lines.

Writes figures/fig_method_overview_2.{png,pdf,svg}.
Assets: bakeoff/method_overview/assets/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

FIG_DIR = Path(__file__).resolve().parent
ASSETS = FIG_DIR / "bakeoff" / "method_overview" / "assets"
STEM = "fig_method_overview_2"

C_TRAIN = "#0B3D5C"
C_INFER = "#047857"
C_LOOP = "#B45309"
C_INK = "#1F2937"
C_MUTED = "#6B7280"
C_PANEL = "#EEF2F7"
C_LINE = "#CBD5E1"
C_ARROW = "#374151"
C_ZOOM_BG = "#FFFBEB"
C_ZOOM_EDGE = "#D6D3D1"


def _load(name: str) -> np.ndarray:
    return np.asarray(Image.open(ASSETS / name).convert("RGB")) / 255.0


def _round(ax, x, y, w, h, fc, ec, lw=0.9, rs=0.04, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={rs}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=z,
        )
    )


def _arrow(ax, x0, y0, x1, y1, *, color=C_ARROW, lw=1.1, ms=8.5):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=ms,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )


def _hline(ax, x0, x1, y, *, color=C_ARROW, lw=1.1):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=5)


def _vline(ax, x, y0, y1, *, color=C_ARROW, lw=1.1):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, solid_capstyle="butt", zorder=5)


def _img_panel(ax, x, y, s, fname: str, caption: str, *, accent=C_INK, cap_y=None):
    """Image frame + caption on a shared baseline (never drawn on the image).

    ``s`` is the panel width; height follows the asset aspect (3-slice stacks are wider).
    """
    img = _load(fname)
    ih, iw = img.shape[:2]
    aspect = ih / max(iw, 1)
    # fit inside a square budget of side s (prefer full width)
    w = s
    h = s * min(aspect, 1.05)
    # vertically center within the square budget
    y0 = y + (s - h) / 2
    _round(ax, x - 0.02, y0 - 0.02, w + 0.04, h + 0.04, "white", accent, lw=0.7, rs=0.02, z=3)
    ax.imshow(
        img,
        extent=(x, x + w, y0, y0 + h),
        origin="upper",
        interpolation="nearest",
        zorder=4,
        aspect="auto",
    )
    ax.text(
        x + w / 2,
        cap_y if cap_y is not None else (y - 0.16),
        caption,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=C_INK,
        zorder=5,
        linespacing=1.05,
        clip_on=False,
    )
    return x + w / 2, y0 + h / 2


def _proc_box(ax, x, y, w, h, text: str, *, fc=C_PANEL, ec=C_INK, tc=C_INK, fs=7.0):
    _round(ax, x, y, w, h, fc, ec, lw=0.85, rs=0.05, z=3)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold",
        color=tc,
        zorder=4,
        linespacing=1.12,
    )
    return x + w / 2, y + h / 2


def _math_note(ax, x, y, text: str, *, color=C_MUTED, fs=5.8, ha="center", va="top"):
    """Secondary equation annotation beside / under a main block (not a block label)."""
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color,
        zorder=7,
        linespacing=1.15,
        clip_on=False,
    )


def draw() -> None:
    # Vertical bands (bottom → top), carefully non-overlapping:
    #   footnote | refinement strip | gap | inference imgs+caps | math notes | loop | banner
    #   | divider | training imgs+caps | math notes | banner
    fig_w, fig_h = 12.6, 7.05
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    banner_h = 0.32

    # ===================== TRAINING (content + banner centered) =====================
    s = 0.88
    y_img_t = 5.20
    y_mid_t = y_img_t + s / 2
    cap_y_t = y_img_t - 0.18

    train_items = [
        ("img", "healthy_patch.png", "Healthy 64³\n(3 slices)"),
        ("img", "noise_patch.png", "Noise volume"),
        ("box", "3D Rectified\nFlow Network", dict(fc="#D9E8F2", ec=C_TRAIN, tc=C_TRAIN, w=1.75, h=0.78)),
        ("box", "Velocity Loss", dict(fc=C_PANEL, ec=C_INK, tc=C_INK, w=1.25, h=0.66)),
        ("img", "learned_prior.png", "Learned healthy\nprior (3D)"),
    ]
    widths = [s if k == "img" else b["w"] for k, a, b in train_items]
    gap_t = 0.30
    total_t = sum(widths) + gap_t * (len(widths) - 1)
    x0_t = (fig_w - total_t) / 2

    y_train_banner = 6.55
    banner_pad = 0.55
    train_banner_w = total_t + banner_pad
    train_banner_x = (fig_w - train_banner_w) / 2
    ax.add_patch(
        Rectangle(
            (train_banner_x, y_train_banner),
            train_banner_w,
            banner_h,
            facecolor=C_TRAIN,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.text(
        fig_w / 2,
        y_train_banner + banner_h / 2,
        "Training  (offline)",
        ha="center",
        va="center",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        zorder=3,
    )

    x = x0_t
    centers_t, lefts_t, rights_t = [], [], []
    for (kind, a, b), w in zip(train_items, widths):
        if kind == "img":
            accent = C_TRAIN if "prior" in b.lower() else C_INK
            cx, cy = _img_panel(ax, x, y_img_t, s, a, b, accent=accent, cap_y=cap_y_t)
        else:
            cx, cy = _proc_box(ax, x, y_mid_t - b["h"] / 2, b["w"], b["h"], a, fc=b["fc"], ec=b["ec"], tc=b["tc"])
        centers_t.append((cx, cy))
        lefts_t.append(x)
        rights_t.append(x + w)
        x += w + gap_t
    for i in range(len(centers_t) - 1):
        _arrow(ax, rights_t[i] + 0.04, y_mid_t, lefts_t[i + 1] - 0.04, y_mid_t)

    # Math notes under training (RF path + velocity objective; paper notation)
    y_math_t = cap_y_t - 0.38
    _math_note(
        ax,
        centers_t[1][0],
        y_math_t,
        r"$\boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$",
        color=C_TRAIN,
        fs=5.8,
    )
    _math_note(
        ax,
        centers_t[2][0],
        y_math_t,
        r"$\mathbf{x}_t=(1-t/T)\mathbf{x}_0+(t/T)\boldsymbol{\varepsilon}$"
        "\n"
        r"$\mathbf{u}(\mathbf{x}_t,t)=\mathbf{x}_0-\boldsymbol{\varepsilon}$",
        color=C_TRAIN,
        fs=5.4,
    )
    _math_note(
        ax,
        centers_t[3][0],
        y_math_t,
        r"$\mathcal{L}(\theta)=\mathbb{E}[\|v_\theta(\mathbf{x}_t,t)-\mathbf{u}\|_1]$",
        color=C_TRAIN,
        fs=5.4,
    )

    # short solid marker: prior is used at inference (no long dashed curve)
    y_div = 4.55
    px = centers_t[-1][0]
    _vline(ax, px, y_math_t - 0.22, y_div + 0.02, color=C_TRAIN, lw=1.15)
    _arrow(ax, px, y_div + 0.02, px, y_div - 0.01, color=C_TRAIN, lw=1.15, ms=7.5)
    ax.text(
        px + 0.10,
        (y_math_t - 0.10 + y_div) / 2,
        "→ inference",
        ha="left",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=C_TRAIN,
        zorder=7,
    )

    # ===================== INFERENCE (content + banner centered) =====================
    si = 0.78
    y_img_i = 2.55
    y_mid_i = y_img_i + si / 2
    cap_y_i = y_img_i - 0.17

    infer_items = [
        ("img", "test_ct.png", "CT 64³\n(3 slices)"),
        ("img", "box_prompt.png", "Box prompt"),
        ("img", "initial_hole.png", "Initial hole"),
        (
            "box",
            "Mask-guided RF\ninpainting\n(uses prior)",
            dict(fc="#D8F3E8", ec=C_INFER, tc=C_INFER, w=1.60, h=0.90),
        ),
        ("img", "healthy_reconstruction.png", "Healthy recon."),
        ("img", "residual.png", r"Residual $|\mathbf{x}_0-\hat{\mathbf{x}}|$"),
        ("box", "Threshold\n+ refine", dict(fc=C_PANEL, ec=C_INK, tc=C_INK, w=1.15, h=0.68)),
        ("img", "final_seg.png", "Final (red / lime)"),
    ]
    widths_i = [si if k == "img" else b["w"] for k, a, b in infer_items]
    gap_i = 0.15
    total_i = sum(widths_i) + gap_i * (len(widths_i) - 1)
    x0_i = (fig_w - total_i) / 2

    y_inf_banner = 4.20
    inf_banner_w = total_i + 0.40
    inf_banner_x = (fig_w - inf_banner_w) / 2
    ax.add_patch(
        Rectangle(
            (inf_banner_x, y_inf_banner),
            inf_banner_w,
            banner_h,
            facecolor=C_INFER,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.text(
        fig_w / 2,
        y_inf_banner + banner_h / 2,
        "Inference  (online)",
        ha="center",
        va="center",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        zorder=3,
    )

    # centered divider spanning the wider of train/infer content
    div_w = max(total_t, total_i) + 0.30
    div_x0 = (fig_w - div_w) / 2
    ax.plot([div_x0, div_x0 + div_w], [y_div, y_div], color=C_LINE, lw=1.05, zorder=2)

    x = x0_i
    centers_i, lefts_i, rights_i = [], [], []
    hole_idx, update_idx = 2, 6
    for idx, ((kind, a, b), w) in enumerate(zip(infer_items, widths_i)):
        if kind == "img":
            accent = C_INFER if idx == len(infer_items) - 1 else C_INK
            cx, cy = _img_panel(ax, x, y_img_i, si, a, b, accent=accent, cap_y=cap_y_i)
        else:
            cx, cy = _proc_box(
                ax, x, y_mid_i - b["h"] / 2, b["w"], b["h"], a, fc=b["fc"], ec=b["ec"], tc=b["tc"], fs=6.5
            )
        centers_i.append((cx, cy))
        lefts_i.append(x)
        rights_i.append(x + w)
        x += w + gap_i
    for i in range(len(centers_i) - 1):
        _arrow(ax, rights_i[i] + 0.03, y_mid_i, lefts_i[i + 1] - 0.03, y_mid_i)

    # Compact must-show math (paper notation) under inference blocks
    y_math_i = cap_y_i - 0.42
    _math_note(
        ax,
        centers_i[1][0],
        y_math_i,
        r"$\rightarrow\,\mathbf{h}^{(0)}$",
        color=C_INFER,
        fs=5.3,
    )
    _math_note(
        ax,
        centers_i[2][0],
        y_math_i,
        r"$\mathbf{h}^{(0)}_z=\mathrm{AABB}_{2D}(\mathbf{y}_z)\oplus p$",
        color=C_INFER,
        fs=5.1,
    )
    _math_note(
        ax,
        centers_i[3][0],
        y_math_i,
        r"$\mathbf{x}_t\!\leftarrow\!\mathbf{m}\odot\tilde{\mathbf{x}}_t(\mathbf{x}_0)+(1-\mathbf{m})\odot\mathbf{x}_t$"
        "\n"
        r"$\mathbf{x}_t\!\leftarrow\!\mathbf{x}_t+v_\theta(\mathbf{x}_t,t)\,\Delta t$",
        color=C_INFER,
        fs=4.9,
    )
    _math_note(
        ax,
        centers_i[4][0],
        y_math_i,
        r"$\hat{\mathbf{x}}^{(r)}$",
        color=C_INFER,
        fs=5.4,
    )
    _math_note(
        ax,
        centers_i[5][0],
        y_math_i,
        r"$\mathbf{d}^{(r)}=|\mathbf{x}_0-\hat{\mathbf{x}}^{(r)}|$",
        color=C_LOOP,
        fs=5.2,
    )
    _math_note(
        ax,
        centers_i[6][0],
        y_math_i,
        r"$\mathbf{h}^{(r+1)}=\mathbf{1}[\mathbf{d}^{(r)}>\tau]$"
        "\n"
        r"$\odot\,\mathrm{Dilate}(\mathbf{h}^{(r)};\rho)$",
        color=C_LOOP,
        fs=5.0,
    )
    _math_note(
        ax,
        centers_i[7][0],
        y_math_i,
        r"$\hat{\mathbf{y}}=\mathbf{h}^{(R)}$",
        color=C_INFER,
        fs=5.5,
    )

    # iterate loop in the clear band BETWEEN image tops and inference banner
    x_hole, x_upd = centers_i[hole_idx][0], centers_i[update_idx][0]
    y_top_img = y_img_i + si
    y_loop = 0.5 * (y_top_img + y_inf_banner) - 0.04
    _vline(ax, x_upd, y_top_img + 0.04, y_loop, color=C_LOOP, lw=1.15)
    _hline(ax, x_upd, x_hole, y_loop, color=C_LOOP, lw=1.15)
    _arrow(ax, x_hole, y_loop, x_hole, y_top_img + 0.04, color=C_LOOP, lw=1.15, ms=7.5)
    ax.text(
        (x_hole + x_upd) / 2,
        y_loop - 0.04,
        r"iterate $r=0,\ldots,R$",
        ha="center",
        va="top",
        fontsize=6.7,
        fontweight="bold",
        color=C_LOOP,
        zorder=7,
    )

    # ===================== MASK REFINEMENT (horizontal, centered) =====================
    y_strip, strip_h = 0.12, 1.55
    zoom = [
        ("initial_mask.png", "Initial mask"),
        ("iter1.png", "Iteration 1"),
        ("iter2.png", "Iteration 2"),
    ]
    sz = 0.90
    gap_z = 0.70
    total_z = len(zoom) * sz + (len(zoom) - 1) * gap_z
    pad_z = 0.35
    strip_w = total_z + 2 * pad_z
    strip_x = (fig_w - strip_w) / 2
    _round(ax, strip_x, y_strip, strip_w, strip_h, C_ZOOM_BG, C_ZOOM_EDGE, lw=0.85, rs=0.05, z=2)
    ax.text(
        strip_x + strip_w / 2,
        y_strip + strip_h - 0.10,
        "Mask refinement",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=C_INK,
        zorder=5,
    )
    # refinement idea equation beside the strip title
    _math_note(
        ax,
        strip_x + strip_w / 2,
        y_strip + strip_h - 0.28,
        r"$\mathbf{h}^{(r+1)}=\mathbf{1}[\mathbf{d}^{(r)}>\tau]\odot\mathrm{Dilate}(\mathbf{h}^{(r)};\rho),"
        r"\ \ \hat{\mathbf{y}}=\mathbf{h}^{(R)}$",
        color=C_LOOP,
        fs=5.7,
    )

    xz0 = strip_x + pad_z
    yz = y_strip + 0.10
    for i, (fn, lab) in enumerate(zoom):
        xz = xz0 + i * (sz + gap_z)
        ax.text(
            xz + sz / 2,
            yz + sz + 0.02,
            lab,
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            color=C_INK,
            zorder=5,
        )
        _round(ax, xz - 0.018, yz - 0.018, sz + 0.036, sz + 0.036, "white", C_INK, lw=0.6, rs=0.02, z=3)
        ax.imshow(
            _load(fn),
            extent=(xz, xz + sz, yz, yz + sz),
            origin="upper",
            interpolation="nearest",
            zorder=4,
            aspect="auto",
        )
        if i < len(zoom) - 1:
            _arrow(
                ax,
                xz + sz + 0.08,
                yz + sz / 2,
                xz + sz + gap_z - 0.08,
                yz + sz / 2,
                color=C_LOOP,
                lw=1.05,
                ms=7.5,
            )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.01)
    for ext in ("png", "pdf", "svg"):
        out = FIG_DIR / f"{STEM}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.05)
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    draw()
