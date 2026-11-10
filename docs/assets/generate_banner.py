#!/usr/bin/env python3
"""Banner for robotac-auto-max, drawn from the run's real native PCD."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2400, 640
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_DIR = Path("/usr/share/matplotlib/mpl-data/fonts/ttf")
WHITE = (248, 250, 252)
CYAN = (34, 211, 238)

PCD = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "dev_ws/midterm/runs/auto_max_20261109_205614/auto_max_20261109_205614_native.pcd")
NPZ = Path(__file__).with_name("banner_points.npz")


def f_sans(size):
    return ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), size)


def f_mono(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf")), size)


def load_points(path):
    raw = path.read_bytes()
    marker = raw.index(b"DATA binary\n") + len(b"DATA binary\n")
    pts = np.frombuffer(raw[marker:], dtype="<f4").reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "banner.png")
    if PCD.exists():
        pts = load_points(PCD)
        total = len(pts)
        rng = np.random.default_rng(20261109)
        floor = pts[pts[:, 2] < 0.15]
        walls = pts[pts[:, 2] >= 0.15]
        keep = np.concatenate([
            rng.choice(len(floor), size=min(30000, len(floor)), replace=False),
            rng.choice(len(walls), size=min(90000, len(walls)), replace=False),
        ])
        pts = pts[keep]
    else:
        data = np.load(NPZ)
        pts = data["pts"].astype(np.float32) * float(data["scale"])
        total = int(data["total"])

    # background
    t = np.linspace(0, 1, H)[:, None, None]
    img = np.repeat(np.array((5, 9, 17))[None, None, :] * (1 - t)
                    + np.array((11, 22, 38))[None, None, :] * t, W, axis=1)
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    img[((xs % 48) < 1) | ((ys % 48) < 1)] += 9.0
    canvas = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    # fit the map into the right panel (keep aspect)
    x0, x1, y0, y1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    bx0, bx1, by0, by1 = 1430, 2320, 70, 585
    scale = min((bx1 - bx0) / (x1 - x0), (by1 - by0) / (y1 - y0))
    ox = (bx0 + bx1) / 2 - (x0 + x1) / 2 * scale
    oy = (by0 + by1) / 2 + (y0 + y1) / 2 * scale

    d = ImageDraw.Draw(canvas, "RGBA")
    px = pts[:, 0] * scale + ox
    py = -pts[:, 1] * scale + oy
    z = np.clip(pts[:, 2] / 3.0, 0, 1)
    order = np.argsort(z)  # draw floor first, structure on top
    for i in order:
        r = 1.15
        if z[i] < 0.05:
            col = (80, 170, 200, 34)
        else:
            a = int(90 + 110 * z[i])
            col = (int(90 + 100 * z[i]), int(190 + 50 * z[i]), 235, a)
        d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=col)

    # frame + annotation
    d.rectangle([bx0 - 14, by0 - 14, bx1 + 14, by1 + 14], outline=CYAN + (28,), width=1)
    d.text((bx1 + 14, by1 + 26), f"NATIVE PCD  ·  {total:,} PTS", font=f_mono(15), fill=(100, 116, 139, 255), anchor="ra")
    d.text((bx0 - 14, by1 + 26), "RUN 2026-11-09  ·  12 / 12 QA", font=f_mono(15), fill=(100, 116, 139, 255))

    # left text
    accent = CYAN
    left = 140
    d.rectangle((96, 142, 104, 552), fill=accent + (170,))
    d.text((left, 136), "ROBOTAC  //  AUTONOMOUS  MAPPING", font=f_mono(26), fill=accent + (255,))
    d.text((left - 4, 168), "Autonomous", font=f_sans(96), fill=WHITE + (255,))
    d.text((left - 4, 282), "Mapping", font=f_sans(96), fill=accent + (255,))
    d.text((left, 430), "UeSim + MuJoCo + ROS 2 Humble   ·   odometry-anchored voxel fusion",
           font=f_mono(27), fill=(148, 163, 184, 255))
    x = left
    for value, label in (("30", "MIN BUDGET"), ("0.45", "M/S LIMIT"), ("2.5", "CM VOXEL"), ("12", "QA CHECKS")):
        d.text((x, 498), value, font=f_mono(30, True), fill=WHITE + (255,))
        wv = d.textlength(value, font=f_mono(30, True))
        d.text((x + wv + 10, 508), label, font=f_mono(19), fill=(100, 116, 139, 255))
        x += wv + 10 + d.textlength(label, font=f_mono(19)) + 42

    canvas.convert("RGB").save(out, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
