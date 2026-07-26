#!/usr/bin/env python3
"""Export method-overview v2 as a self-contained diagrams.net (.drawio) file.

Embeds LIDC sample PNGs from bakeoff/method_overview/assets/.
Outputs:
  figures/fig_method_overview_2.drawio
  figures/bakeoff/method_overview/fig_method_overview_2.drawio
"""
from __future__ import annotations

import base64
from pathlib import Path
from xml.sax.saxutils import escape

FIG = Path(__file__).resolve().parent
ASSETS = FIG / "bakeoff" / "method_overview" / "assets"
OUTS = [
    FIG / "fig_method_overview_2.drawio",
    FIG / "bakeoff" / "method_overview" / "fig_method_overview_2.drawio",
]

PW, PH = 1280, 720


def b64(name: str) -> str:
    return "data:image/png," + base64.b64encode((ASSETS / name).read_bytes()).decode("ascii")


def cell(cid: str, value: str, style: str, x: float, y: float, w: float, h: float) -> str:
    # draw.io uses &#xa; for newlines inside attribute values
    val = escape(value).replace("\n", "&#xa;")
    return (
        f'    <mxCell id="{cid}" value="{val}" style="{style}" '
        f'vertex="1" parent="1">\n'
        f'      <mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/>\n'
        f'    </mxCell>\n'
    )


def edge(cid: str, source: str, target: str, style: str, points=None) -> str:
    if points:
        pts = "\n".join(f'          <mxPoint x="{x:.1f}" y="{y:.1f}"/>' for x, y in points)
        geo = (
            '      <mxGeometry relative="1" as="geometry">\n'
            f'        <Array as="points">\n{pts}\n        </Array>\n'
            "      </mxGeometry>\n"
        )
    else:
        geo = '      <mxGeometry relative="1" as="geometry"/>\n'
    return (
        f'    <mxCell id="{cid}" value="" style="{style}" edge="1" parent="1" '
        f'source="{source}" target="{target}">\n{geo}    </mxCell>\n'
    )


def img_style(name: str) -> str:
    return (
        "shape=image;html=1;verticalLabelPosition=bottom;verticalAlign=top;"
        "imageAspect=1;aspect=fixed;fontSize=11;fontStyle=1;fontColor=#1F2937;"
        f"spacingTop=4;image={b64(name)}"
    )


def main() -> None:
    arrow = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=1;endArrow=block;endFill=1;strokeColor=#374151;strokeWidth=1.5;"
        "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
    )
    loop_arrow = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
        "html=1;endArrow=block;endFill=1;strokeColor=#B45309;strokeWidth=1.6;"
        "exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
    )
    z_arrow = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;"
        "strokeColor=#B45309;strokeWidth=1.5;"
        "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
    )

    parts: list[str] = ['    <mxCell id="0"/>\n', '    <mxCell id="1" parent="0"/>\n']

    # Training (LIDC #1370 stacks) / Inference (LIDC #1183 stacks)
    train_specs = [
        ("t_healthy", "img", "healthy_patch.png", "Healthy 64³\n(3 slices)", 90),
        ("t_noise", "img", "noise_patch.png", "Noise volume", 90),
        ("t_rf", "box", None, "3D Rectified\nFlow Network", 150),
        ("t_loss", "box", None, "Velocity Loss", 120),
        ("t_prior", "img", "learned_prior.png", "Learned healthy\nprior (3D)", 90),
    ]
    gap, ty, img = 36, 80, 90
    tw = sum(s[4] for s in train_specs) + gap * (len(train_specs) - 1)
    tx = (PW - tw) / 2

    parts.append(
        cell(
            "tbanner",
            "Training  (offline)",
            "rounded=1;whiteSpace=wrap;html=1;fillColor=#0B3D5C;strokeColor=none;"
            "fontColor=#FFFFFF;fontSize=14;fontStyle=1;align=center;verticalAlign=middle;arcSize=6;",
            (PW - (tw + 40)) / 2,
            24,
            tw + 40,
            34,
        )
    )

    x = tx
    train_ids = []
    for cid, kind, fname, label, w in train_specs:
        train_ids.append(cid)
        if kind == "img":
            parts.append(cell(cid, label, img_style(fname), x, ty, w, w))
        else:
            style = (
                "rounded=1;whiteSpace=wrap;html=1;fillColor=#D9E8F2;strokeColor=#0B3D5C;"
                "fontColor=#0B3D5C;fontSize=12;fontStyle=1;align=center;verticalAlign=middle;arcSize=10;"
                if "Network" in label
                else "rounded=1;whiteSpace=wrap;html=1;fillColor=#EEF2F7;strokeColor=#1F2937;"
                "fontColor=#1F2937;fontSize=12;fontStyle=1;align=center;verticalAlign=middle;arcSize=10;"
            )
            parts.append(cell(cid, label, style, x, ty + (img - 70) / 2, w, 70))
        x += w + gap

    for i in range(len(train_ids) - 1):
        parts.append(edge(f"te{i}", train_ids[i], train_ids[i + 1], arrow))

    parts.append(
        cell(
            "prior_note",
            "→ inference",
            "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
            "fontSize=11;fontStyle=1;fontColor=#0B3D5C;",
            tx + tw - 100,
            ty + img + 28,
            100,
            20,
        )
    )
    parts.append(
        '    <mxCell id="prior_drop" value="" style="'
        "endArrow=block;endFill=1;strokeColor=#0B3D5C;strokeWidth=1.4;html=1;"
        '" edge="1" parent="1" source="t_prior">\n'
        '      <mxGeometry relative="1" as="geometry">\n'
        f'        <mxPoint x="{tx + tw - 45:.1f}" y="230" as="targetPoint"/>\n'
        "      </mxGeometry>\n"
        "    </mxCell>\n"
    )
    parts.append(cell("divider", "", "line;strokeWidth=1.2;strokeColor=#CBD5E1;", 80, 240, PW - 160, 10))

    # Inference
    infer_specs = [
        ("i_ct", "img", "test_ct.png", "CT 64³\n(3 slices)", 78),
        ("i_box", "img", "box_prompt.png", "Box prompt", 78),
        ("i_hole", "img", "initial_hole.png", "Initial hole", 78),
        ("i_inpaint", "box", None, "Mask-guided RF\ninpainting\n(uses prior)", 140),
        ("i_recon", "img", "healthy_reconstruction.png", "Healthy recon.", 78),
        ("i_resid", "img", "residual.png", "Residual |x-x̂|", 78),
        ("i_thr", "box", None, "Threshold\n+ refine", 110),
        ("i_final", "img", "final_seg.png", "Final (red / lime)", 78),
    ]
    gap_i, iy = 22, 300
    iw = sum(s[4] for s in infer_specs) + gap_i * (len(infer_specs) - 1)
    ix0 = (PW - iw) / 2

    parts.append(
        cell(
            "ibanner",
            "Inference  (online)",
            "rounded=1;whiteSpace=wrap;html=1;fillColor=#047857;strokeColor=none;"
            "fontColor=#FFFFFF;fontSize=14;fontStyle=1;align=center;verticalAlign=middle;arcSize=6;",
            (PW - (iw + 40)) / 2,
            255,
            iw + 40,
            34,
        )
    )

    x = ix0
    infer_ids = []
    for cid, kind, fname, label, w in infer_specs:
        infer_ids.append(cid)
        if kind == "img":
            parts.append(cell(cid, label, img_style(fname), x, iy, w, w))
        else:
            style = (
                "rounded=1;whiteSpace=wrap;html=1;fillColor=#D8F3E8;strokeColor=#047857;"
                "fontColor=#047857;fontSize=11;fontStyle=1;align=center;verticalAlign=middle;arcSize=10;"
                if "inpainting" in label
                else "rounded=1;whiteSpace=wrap;html=1;fillColor=#EEF2F7;strokeColor=#1F2937;"
                "fontColor=#1F2937;fontSize=11;fontStyle=1;align=center;verticalAlign=middle;arcSize=10;"
            )
            bh = 78 if "inpainting" in label else 60
            parts.append(cell(cid, label, style, x, iy + (78 - bh) / 2, w, bh))
        x += w + gap_i

    for i in range(len(infer_ids) - 1):
        parts.append(edge(f"ie{i}", infer_ids[i], infer_ids[i + 1], arrow))

    # loop above inference
    x_thr = ix0 + sum(s[4] + gap_i for s in infer_specs[:6]) + infer_specs[6][4] / 2
    x_hole = ix0 + 78 + gap_i + 78 / 2
    parts.append(
        edge("loop", "i_thr", "i_hole", loop_arrow, points=[(x_thr, iy - 28), (x_hole, iy - 28)])
    )
    parts.append(
        cell(
            "looptxt",
            "iterate R rounds",
            "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
            "fontSize=11;fontStyle=1;fontColor=#B45309;",
            (x_hole + x_thr) / 2 - 80,
            iy - 52,
            160,
            20,
        )
    )

    # Mask refinement
    zoom = [
        ("z1", "iter1.png", "Iteration 1"),
        ("z2", "residual1.png", "Residual"),
        ("z3", "iter2.png", "Iteration 2"),
        ("z4", "final_zoom.png", "Final"),
    ]
    zs, zgap = 88, 70
    zw = len(zoom) * zs + (len(zoom) - 1) * zgap
    pad = 40
    box_w = zw + 2 * pad
    box_x = (PW - box_w) / 2
    box_y, box_h = 470, 200
    parts.append(
        cell(
            "zbox",
            "",
            "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFBEB;strokeColor=#D6D3D1;arcSize=8;",
            box_x,
            box_y,
            box_w,
            box_h,
        )
    )
    parts.append(
        cell(
            "ztitle",
            "Mask refinement",
            "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
            "fontSize=13;fontStyle=1;fontColor=#1F2937;",
            box_x,
            box_y + 8,
            box_w,
            24,
        )
    )
    zx, zy = box_x + pad, box_y + 55
    zids = []
    for cid, fname, label in zoom:
        zids.append(cid)
        parts.append(cell(cid, label, img_style(fname), zx, zy, zs, zs))
        zx += zs + zgap
    for i in range(len(zids) - 1):
        parts.append(edge(f"ze{i}", zids[i], zids[i + 1], z_arrow))

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-07-26T16:40:00.000Z" '
        'agent="Cursor" version="22.1.0" type="device">\n'
        '  <diagram id="method-overview-v2" name="Method overview v2">\n'
        f'    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" '
        f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        f'pageWidth="{PW}" pageHeight="{PH}" math="0" shadow="0">\n'
        "      <root>\n"
        f"{''.join(parts)}"
        "      </root>\n"
        "    </mxGraphModel>\n"
        "  </diagram>\n"
        "</mxfile>\n"
    )
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(xml)
        print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
