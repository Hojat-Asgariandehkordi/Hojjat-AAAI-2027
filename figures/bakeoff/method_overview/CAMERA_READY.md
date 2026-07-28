# Method overview — camera-ready

**Paper include:** `figures/fig_method_overview_2.pdf`  
(mirrored as `figures/fig_method_overview.pdf`)

**Editable source:** `fig_method_overview_2.drawio` (also under `figures/`)

**Assets:** `assets/` — train LIDC #1370; inference/refinement LIDC #1183  
(40-step R=2; iter-2 Dice > iter-1). Compact must-show equations sit beside blocks.

**Regenerate**
```bash
nninteractive python figures/export_method_overview_tikz_assets.py
nninteractive python figures/make_method_overview_v2.py
# then sync bakeoff mirrors / drawio embeds as needed
```

**Equations in draw.io:** HTML/Unicode math beside blocks (bold-italic vectors, subscripts); no MathJax `$$...$$`. Paper PDF uses `figures/fig_method_overview_2.pdf` from `make_method_overview_v2.py`.
