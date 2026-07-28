Method-overview assets (3 consecutive axial slices per panel).
Training row: LIDC #1370 (z=31, stack [30, 31, 32]).
Inference / refinement: LIDC #1183 (z=25, stack [24, 25, 26]).
Refine Dice (40-step): r0=0.8169, r1=0.9210, r2=0.9286 (Δr2−r1=+0.0076).
Mask strip uses real refine-round holes from method_overview_refine/.
Regenerate: nninteractive python figures/export_method_overview_tikz_assets.py
