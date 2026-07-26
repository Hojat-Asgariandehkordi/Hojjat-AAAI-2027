# Method overview TikZ v2

File: [`fig_method_overview(2).tikz.tex`](fig_method_overview(2).tikz.tex)

Real LIDC `#1370` patches live in [`assets/`](assets/).

## Regenerate assets

```bash
/home/morteza/.conda/envs/nninteractive/bin/python \
  /home/morteza/MortezaStudentsS02/VAE_Conrol_NET/Hojjat-AAAI-2027/figures/export_method_overview_tikz_assets.py
```

## Compile

```bash
cd figures/bakeoff/method_overview
pdflatex "fig_method_overview(2).tikz.tex"
```

Output: `fig_method_overview(2).tikz.pdf`
