# Figure bake-off (pick the best per stem)

**Paper format:** LaTeX (AAAI 2027) — prefer **vector PDF/SVG** in `\includegraphics`.

## Results figures (data)

| Priority | Stem | Recommended for camera-ready | Also generated |
|----------|------|------------------------------|----------------|
| P0 | `fig_results_bars` | **Matplotlib + Seaborn hbar** | seaborn_vbar, Plotly HTML, ggplot2 |
| P0 | `fig_healthy_fp` | **Seaborn hbar** | seaborn_vbar, Plotly, ggplot2 |
| P1 | `fig_healthy_fp_tradeoff` | **Seaborn 2-panel** | seaborn_scatter, Plotly, ggplot2 |
| P1 | `fig_benchmark_dice_3datasets` | **Seaborn grouped** | seaborn_facet, Plotly, ggplot2 |
| P1 | `fig_ablation_steps` | **Seaborn hbar** | seaborn_vbar, Plotly, ggplot2 |

Layout:

```text
bakeoff/<stem>/<tool>/{pdf,svg,png,html}
bakeoff/_csv/                 # shared CSVs for ggplot2 / replot
bakeoff/method_overview/      # Mermaid / DOT / TikZ / D2 sources
```

### Regenerate

```bash
# Python (Seaborn + Plotly)
/home/morteza/.conda/envs/ddpm1/bin/python bakeoff/generate_bakeoff.py

# R (ggplot2) — installs packages on first run if missing
Rscript bakeoff/ggplot2_bakeoff.R
```

**Plotly** = interactive HTML for supplements (`*_interactive` / `bakeoff/*/plotly/*.html`).  
**ggplot2** = optional if your lab prefers R; same numbers via `_csv/`.

## Schematics (method overview)

| Tool | File | Best for |
|------|------|----------|
| draw.io (current) | `../fig_method_overview.drawio` | Camera-ready polish |
| Mermaid | `method_overview/fig_method_overview.mmd` | Fast draft / Markdown preview |
| Graphviz | `method_overview/fig_method_overview.dot` | Clean pipeline graphs |
| TikZ | `method_overview/fig_method_overview.tikz.tex` | Native LaTeX fonts |
| D2 | `method_overview/fig_method_overview.d2` | Modern architecture SVG |
| Excalidraw | *(manual)* | Sketch look only — skip for AAAI |

### Render schematics

```bash
# Graphviz
dot -Tpdf -Tsvg -o method_overview/fig_method_overview.pdf method_overview/fig_method_overview.dot

# TikZ
pdflatex -output-directory method_overview method_overview/fig_method_overview.tikz.tex

# D2 (if installed)
d2 method_overview/fig_method_overview.d2 method_overview/fig_method_overview_d2.svg
```

## How to choose

1. Open each `bakeoff/<stem>/*/fig_*.png` side by side.
2. Prefer **horizontal bars** for ranked methods (AAAI single/double column).
3. Keep **one** camera-ready toolchain for all quantitative figures (Seaborn) so palette/fonts match.
4. Use **Plotly HTML** only in supplement / reviewer browsing.
5. For the method figure: start from Mermaid/DOT to lock structure, then finish in **draw.io** or **TikZ**.
