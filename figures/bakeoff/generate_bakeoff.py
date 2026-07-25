#!/usr/bin/env python3
"""Generate multi-toolchain bake-off outputs for AAAI results figures.

Paper format: LaTeX (AAAI). Data figures → Matplotlib/Seaborn + Plotly (+ CSV for ggplot2).
Schematics live under bakeoff/method_overview/ (Mermaid / Graphviz / TikZ / D2).

Outputs: figures/bakeoff/<stem>/<tool>/{png,pdf,svg,html}
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

ROOT = Path(__file__).resolve().parents[1]
SNAP = json.loads((ROOT / "figure_data_snapshot.json").read_text())
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

AAAI = dict(single_w=3.35, double_w=6.90, dpi=300)
C = {
    "ours": "#0B3D5C",
    "sam2d": "#4B5563",
    "nni": "#6B7280",
    "segvol": "#C2410C",
    "vista": "#9A3412",
    "sam3d": "#78716C",
    "edge": "#111827",
    "spine": "#1F2937",
    "grid": "#E5E7EB",
    "lidc": "#0B3D5C",
    "maisi": "#B45309",
    "nsclc": "#047857",
}
METHOD_COLORS = {
    "SAM-Med2D": C["sam2d"],
    "Ours": C["ours"],
    "Ours (40-step, R=2)": C["ours"],
    "Ours (60-step, R=3)": C["ours"],
    "nnInteractive": C["nni"],
    "SegVol": C["segvol"],
    "VISTA3D": C["vista"],
    "SAM-Med3D": C["sam3d"],
}
METHOD_ORDER = ["SAM-Med2D", "Ours", "nnInteractive", "SAM-Med3D", "SegVol", "VISTA3D"]
DATASET_COLORS = {"LIDC": C["lidc"], "MAISI": C["maisi"], "NSCLC": C["nsclc"]}


def cut_dp(x, n=2):
    a = np.asarray(x, dtype=float)
    return np.trunc(a * 10**n) / 10**n


def fmt_cut(v, n=2):
    return f"{cut_dp(v, n):.{n}f}"


def apply_style():
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "figure.dpi": AAAI["dpi"],
            "savefig.dpi": AAAI["dpi"],
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": C["spine"],
            "axes.linewidth": 0.8,
            "grid.color": C["grid"],
            "grid.linestyle": ":",
        }
    )


def dest(stem: str, tool: str) -> Path:
    d = OUT / stem / tool
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_mpl(fig, d: Path, stem: str):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(d / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(" ", d.relative_to(OUT), "→ pdf/svg/png")


def hbar(ax, labels, values, *, colors, xlabel, xlim=None, value_fmt="cut2", highlight="Ours"):
    y = np.arange(len(labels))
    lab_r, val_r, col_r = list(labels)[::-1], list(values)[::-1], list(colors)[::-1]
    bars = ax.barh(y, val_r, color=col_r, edgecolor=C["edge"], linewidth=0.45, height=0.72, zorder=3)
    for bar, lab in zip(bars, lab_r):
        if highlight and highlight in str(lab):
            bar.set_hatch("///")
    ax.set_yticks(y)
    ax.set_yticklabels(lab_r)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.xaxis.grid(True, linestyle=":", alpha=0.75, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    xmax = xlim[1] if xlim else max(val_r) * 1.14
    for bar, v in zip(bars, val_r):
        label = f"{v:.0f}" if value_fmt == "int" else fmt_cut(v, 2)
        ax.text(
            min(v + 0.012 * xmax, xmax * 0.985),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=7,
            color=C["spine"],
        )


def vbar(ax, labels, values, *, colors, ylabel, ylim=None, value_fmt="cut2", highlight="Ours"):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor=C["edge"], linewidth=0.45, width=0.72, zorder=3)
    for bar, lab in zip(bars, labels):
        if highlight and highlight in str(lab):
            bar.set_hatch("///")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle=":", alpha=0.75, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    ymax = ylim[1] if ylim else max(values) * 1.14
    for bar, v in zip(bars, values):
        label = f"{v:.0f}" if value_fmt == "int" else fmt_cut(v, 2)
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01 * ymax, label, ha="center", va="bottom", fontsize=6.5)


# ---------------------------------------------------------------------------
# Data frames
# ---------------------------------------------------------------------------
df_bench = pd.DataFrame(SNAP["bench"])
df_fp = pd.DataFrame(SNAP["healthy_fp"])
df_abl = pd.DataFrame(SNAP["ablation"])
df_lidc = pd.DataFrame(SNAP["lidc"])

# alias Ours dice for tradeoff labels used in paper plots
dice_map = dict(zip(df_lidc["method"], df_lidc["dice"]))
dice_map["Ours (40-step, R=2)"] = dice_map.get("Ours", float("nan"))


def write_csvs():
    csv_dir = OUT / "_csv"
    csv_dir.mkdir(exist_ok=True)
    df_bench.to_csv(csv_dir / "bench.csv", index=False)
    df_fp.to_csv(csv_dir / "healthy_fp.csv", index=False)
    df_abl.to_csv(csv_dir / "ablation.csv", index=False)
    df_lidc.to_csv(csv_dir / "lidc.csv", index=False)
    trade = []
    for _, r in df_fp.iterrows():
        m = r["method"]
        d = dice_map.get(m, dice_map.get("Ours") if m.startswith("Ours") else float("nan"))
        trade.append({"method": m, "fp_volume": r["volume_mm3"], "lidc_dice": d})
    pd.DataFrame(trade).to_csv(csv_dir / "tradeoff.csv", index=False)
    print("CSVs →", csv_dir.relative_to(OUT))


# ---------------------------------------------------------------------------
# fig_results_bars
# ---------------------------------------------------------------------------
def bake_results_bars():
    stem = "fig_results_bars"
    common = ["SAM-Med2D", "Ours (40-step, R=2)", "nnInteractive", "SAM-Med3D", "SegVol"]
    fp_map = dict(zip(df_fp["method"], df_fp["volume_mm3"]))
    labels = [m for m in common if m in dice_map and m in fp_map]
    dice = [dice_map[m] for m in labels]
    vols = [fp_map[m] for m in labels]
    cols = [METHOD_COLORS.get(m, "#6B7280") for m in labels]

    # Seaborn horizontal (camera-ready default)
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(AAAI["double_w"], 2.75))
    hbar(axes[0], labels, dice, colors=cols, xlabel="Mean Dice ↑", xlim=(0, 0.85))
    axes[0].set_title("(a) True nodules (LIDC)", loc="left")
    hbar(axes[1], labels, vols, colors=cols, xlabel="Pred. volume (mm³) ↓", xlim=(0, max(vols) * 1.18), value_fmt="int")
    axes[1].set_title("(b) Healthy random boxes", loc="left")
    fig.tight_layout()
    save_mpl(fig, dest(stem, "seaborn_hbar"), stem)

    # Seaborn vertical alternative
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(AAAI["double_w"], 2.9))
    vbar(axes[0], labels, dice, colors=cols, ylabel="Mean Dice ↑", ylim=(0, 0.85))
    axes[0].set_title("(a) True nodules (LIDC)", loc="left")
    vbar(axes[1], labels, vols, colors=cols, ylabel="Pred. volume (mm³) ↓", ylim=(0, max(vols) * 1.18), value_fmt="int")
    axes[1].set_title("(b) Healthy random boxes", loc="left")
    fig.tight_layout()
    save_mpl(fig, dest(stem, "seaborn_vbar"), stem)

    if HAS_PLOTLY:
        d = dest(stem, "plotly")
        fig = make_subplots(rows=1, cols=2, subplot_titles=["(a) LIDC Dice", "(b) Healthy FP volume"])
        short = [m.replace("Ours (40-step, R=2)", "Ours") for m in labels]
        fig.add_trace(
            go.Bar(x=cut_dp(dice), y=short, orientation="h", marker_color=cols, text=[fmt_cut(v) for v in dice], textposition="outside", name="Dice"),
            1,
            1,
        )
        fig.add_trace(
            go.Bar(x=vols, y=short, orientation="h", marker_color=cols, text=[f"{v:.0f}" for v in vols], textposition="outside", name="Volume", showlegend=False),
            1,
            2,
        )
        fig.update_layout(template="plotly_white", height=380, width=960, title="Results bars (interactive)", margin=dict(l=120, r=40, t=70, b=40))
        fig.update_yaxes(autorange="reversed")
        fig.write_html(str(d / f"{stem}.html"), include_plotlyjs="cdn")
        try:
            fig.write_image(str(d / f"{stem}.png"), scale=2)
            fig.write_image(str(d / f"{stem}.pdf"))
        except Exception as e:
            print("  plotly static export skipped:", e)
        print(" ", d.relative_to(OUT), "→ html (+static if kaleido)")


def bake_healthy_fp():
    stem = "fig_healthy_fp"
    order = [
        "VISTA3D",
        "Ours (60-step, R=3)",
        "Ours (40-step, R=2)",
        "SegVol",
        "SAM-Med2D",
        "SAM-Med3D",
        "nnInteractive",
    ]
    dfp = df_fp.set_index("method").reindex([m for m in order if m in set(df_fp["method"])]).reset_index()
    labels, vols = dfp["method"].tolist(), dfp["volume_mm3"].tolist()
    cols = [METHOD_COLORS.get(m, "#9CA3AF" if m == "VISTA3D" else "#6B7280") for m in labels]

    apply_style()
    fig, ax = plt.subplots(figsize=(AAAI["single_w"], 2.85))
    hbar(ax, labels, vols, colors=cols, xlabel="Predicted volume (mm³) ↓", xlim=(0, max(vols) * 1.2), value_fmt="int")
    ax.set_title("Healthy FP probe", loc="left")
    fig.tight_layout()
    save_mpl(fig, dest(stem, "seaborn_hbar"), stem)

    apply_style()
    fig, ax = plt.subplots(figsize=(AAAI["double_w"] * 0.85, 2.6))
    vbar(ax, labels, vols, colors=cols, ylabel="Predicted volume (mm³) ↓", ylim=(0, max(vols) * 1.2), value_fmt="int")
    ax.set_title("Healthy FP probe", loc="left")
    fig.tight_layout()
    save_mpl(fig, dest(stem, "seaborn_vbar"), stem)

    if HAS_PLOTLY:
        d = dest(stem, "plotly")
        fig = px.bar(
            dfp.assign(vol=dfp["volume_mm3"]),
            x="volume_mm3",
            y="method",
            orientation="h",
            color="method",
            color_discrete_map=METHOD_COLORS,
            title="Healthy FP volume (interactive)",
            labels={"volume_mm3": "Pred. volume (mm³)", "method": ""},
            height=400,
            width=720,
        )
        fig.update_layout(template="plotly_white", showlegend=False, yaxis=dict(autorange="reversed"))
        fig.write_html(str(d / f"{stem}.html"), include_plotlyjs="cdn")
        try:
            fig.write_image(str(d / f"{stem}.png"), scale=2)
        except Exception as e:
            print("  plotly static export skipped:", e)
        print(" ", d.relative_to(OUT), "→ html")


def bake_tradeoff():
    stem = "fig_healthy_fp_tradeoff"
    order = ["VISTA3D", "Ours (40-step, R=2)", "SegVol", "SAM-Med2D", "SAM-Med3D", "nnInteractive"]
    fp_map = dict(zip(df_fp["method"], df_fp["volume_mm3"]))
    rows = []
    for m in order:
        if m not in fp_map:
            continue
        rows.append({"method": m, "fp_volume": float(fp_map[m]), "lidc_dice": float(dice_map.get(m, np.nan))})
    df_trade = pd.DataFrame(rows)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(AAAI["double_w"], 2.85))
    df_vol = df_trade.sort_values("fp_volume")
    cols = [METHOD_COLORS.get(m, "#6B7280") for m in df_vol["method"]]
    hbar(axes[0], df_vol["method"].tolist(), df_vol["fp_volume"].tolist(), colors=cols, xlabel="Pred. volume (mm³) ↓", xlim=(0, max(df_vol["fp_volume"]) * 1.22), value_fmt="int")
    axes[0].set_title("(a) Healthy FP volume", loc="left")
    ax = axes[1]
    for r in df_trade.itertuples():
        ax.scatter(r.fp_volume, r.lidc_dice, s=70, c=METHOD_COLORS.get(r.method, "#6B7280"), edgecolors=C["edge"], zorder=3)
        ax.annotate(r.method.replace("Ours (40-step, R=2)", "Ours").replace("nnInteractive", "nnInt"), (r.fp_volume, r.lidc_dice), fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Healthy FP volume (mm³) ↓")
    ax.set_ylabel("LIDC Dice ↑")
    ax.set_title("(b) Trade-off", loc="left")
    sns.despine(ax=ax)
    fig.tight_layout()
    save_mpl(fig, dest(stem, "seaborn_hbar"), stem)

    # scatter-only alt
    apply_style()
    fig, ax = plt.subplots(figsize=(AAAI["single_w"] * 1.35, 2.85))
    for r in df_trade.itertuples():
        ax.scatter(r.fp_volume, r.lidc_dice, s=90, c=METHOD_COLORS.get(r.method, "#6B7280"), edgecolors=C["edge"], zorder=3)
        ax.annotate(r.method.replace("Ours (40-step, R=2)", "Ours"), (r.fp_volume, r.lidc_dice), fontsize=6.5, xytext=(5, 3), textcoords="offset points")
    ax.set_xlabel("Healthy FP volume (mm³) ↓")
    ax.set_ylabel("LIDC Dice ↑")
    ax.set_title("Dice vs FP volume", loc="left")
    sns.despine(ax=ax)
    fig.tight_layout()
    save_mpl(fig, dest(stem, "seaborn_scatter"), stem)

    if HAS_PLOTLY:
        d = dest(stem, "plotly")
        fig = px.scatter(
            df_trade,
            x="fp_volume",
            y="lidc_dice",
            text="method",
            color="method",
            color_discrete_map=METHOD_COLORS,
            title="Healthy FP vs LIDC Dice (interactive)",
            labels={"fp_volume": "Healthy FP volume (mm³)", "lidc_dice": "LIDC Dice"},
            height=420,
            width=720,
        )
        fig.update_traces(textposition="top right", marker=dict(size=12))
        fig.update_layout(template="plotly_white", showlegend=False)
        fig.write_html(str(d / f"{stem}.html"), include_plotlyjs="cdn")
        try:
            fig.write_image(str(d / f"{stem}.png"), scale=2)
        except Exception as e:
            print("  plotly static export skipped:", e)
        print(" ", d.relative_to(OUT), "→ html")


def bake_multidataset():
    stem = "fig_benchmark_dice_3datasets"
    df = df_bench.copy()
    df["dice_cut"] = cut_dp(df["dice"])

    apply_style()
    fig, ax = plt.subplots(figsize=(AAAI["double_w"], 3.0))
    sns.barplot(
        data=df,
        y="method",
        x="dice",
        hue="dataset",
        order=METHOD_ORDER,
        hue_order=["LIDC", "MAISI", "NSCLC"],
        palette=DATASET_COLORS,
        ax=ax,
        edgecolor=C["edge"],
        linewidth=0.4,
    )
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Mean Dice ↑")
    ax.set_ylabel("")
    ax.set_title("Matched box prompts · LIDC / MAISI / NSCLC", loc="left")
    ax.legend(title="Dataset", frameon=True, loc="lower right", fontsize=7)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_mpl(fig, dest(stem, "seaborn_grouped"), stem)

    # facet alternative
    apply_style()
    g = sns.catplot(
        data=df,
        kind="bar",
        x="dice",
        y="method",
        col="dataset",
        order=METHOD_ORDER,
        col_order=["LIDC", "MAISI", "NSCLC"],
        color=C["ours"],
        height=2.6,
        aspect=0.85,
        edgecolor=C["edge"],
    )
    g.set_axis_labels("Mean Dice ↑", "")
    g.set(xlim=(0, 0.9))
    for ax in g.axes.flat:
        sns.despine(ax=ax)
    g.savefig(dest(stem, "seaborn_facet") / f"{stem}.pdf", bbox_inches="tight")
    g.savefig(dest(stem, "seaborn_facet") / f"{stem}.svg", bbox_inches="tight")
    g.savefig(dest(stem, "seaborn_facet") / f"{stem}.png", bbox_inches="tight", dpi=AAAI["dpi"])
    plt.close(g.fig)
    print(" ", (dest(stem, "seaborn_facet")).relative_to(OUT), "→ pdf/svg/png")

    if HAS_PLOTLY:
        d = dest(stem, "plotly")
        fig = px.bar(
            df,
            x="dice",
            y="method",
            color="dataset",
            orientation="h",
            barmode="group",
            category_orders={"method": METHOD_ORDER, "dataset": ["LIDC", "MAISI", "NSCLC"]},
            color_discrete_map=DATASET_COLORS,
            title="Multi-dataset Dice (interactive)",
            labels={"dice": "Mean Dice", "method": ""},
            height=420,
            width=900,
        )
        fig.update_layout(template="plotly_white", legend=dict(orientation="h", y=1.08))
        fig.write_html(str(d / f"{stem}.html"), include_plotlyjs="cdn")
        try:
            fig.write_image(str(d / f"{stem}.png"), scale=2)
        except Exception as e:
            print("  plotly static export skipped:", e)
        print(" ", d.relative_to(OUT), "→ html")


def bake_ablation():
    stem = "fig_ablation_steps"
    labels = df_abl["config"].tolist()
    dice = df_abl["dice"].tolist()
    cmap = sns.color_palette("crest", n_colors=len(labels))

    apply_style()
    fig, ax = plt.subplots(figsize=(AAAI["single_w"], 2.55))
    hbar(ax, labels, dice, colors=list(cmap), xlabel="Mean Dice ↑", xlim=(0.70, 0.76), highlight="R=2")
    ax.set_title("Ours ablation (LIDC)", loc="left")
    fig.tight_layout()
    save_mpl(fig, dest(stem, "seaborn_hbar"), stem)

    apply_style()
    fig, ax = plt.subplots(figsize=(AAAI["single_w"] * 1.2, 2.55))
    vbar(ax, labels, dice, colors=list(cmap), ylabel="Mean Dice ↑", ylim=(0.70, 0.76), highlight="R=2")
    ax.set_title("Ours ablation (LIDC)", loc="left")
    fig.tight_layout()
    save_mpl(fig, dest(stem, "seaborn_vbar"), stem)

    if HAS_PLOTLY:
        d = dest(stem, "plotly")
        fig = px.bar(
            df_abl,
            x="dice",
            y="config",
            orientation="h",
            title="Ablation (interactive)",
            labels={"dice": "Mean Dice", "config": ""},
            height=360,
            width=640,
            color="dice",
            color_continuous_scale="Teal",
        )
        fig.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), showlegend=False)
        fig.write_html(str(d / f"{stem}.html"), include_plotlyjs="cdn")
        try:
            fig.write_image(str(d / f"{stem}.png"), scale=2)
        except Exception as e:
            print("  plotly static export skipped:", e)
        print(" ", d.relative_to(OUT), "→ html")


def write_manifest():
    rows = []
    for stem_dir in sorted(p for p in OUT.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name != "method_overview"):
        for tool_dir in sorted(stem_dir.iterdir()):
            if not tool_dir.is_dir():
                continue
            files = sorted(f.name for f in tool_dir.iterdir() if f.is_file())
            rows.append({"stem": stem_dir.name, "tool": tool_dir.name, "files": ", ".join(files)})
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "MANIFEST.csv", index=False)
    print("\nMANIFEST:")
    print(man.to_string(index=False))


def main():
    print("Bake-off root:", OUT)
    write_csvs()
    bake_results_bars()
    bake_healthy_fp()
    bake_tradeoff()
    bake_multidataset()
    bake_ablation()
    write_manifest()
    print("\nDone. Open bakeoff/*/seaborn_*/ and bakeoff/*/plotly/ to compare.")


if __name__ == "__main__":
    main()
