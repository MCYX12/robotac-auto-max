#!/usr/bin/env python3
"""Render docs/assets/banner.png — deterministic, dependency-light (PIL + numpy).

Run: python3 docs/assets/generate_banner.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2400, 640
OUT = Path(__file__).with_name("banner.png")
FONT_DIR = Path("/usr/share/matplotlib/mpl-data/fonts/ttf")
SANS = FONT_DIR / "DejaVuSans.ttf"
SANS_B = FONT_DIR / "DejaVuSans-Bold.ttf"
MONO = FONT_DIR / "DejaVuSansMono.ttf"
MONO_B = FONT_DIR / "DejaVuSansMono-Bold.ttf"

CYAN = np.array([34, 211, 238], dtype=float)
AMBER = np.array([245, 158, 11], dtype=float)
WHITE = np.array([248, 250, 252], dtype=float)


def font(path, size):
    return ImageFont.truetype(str(path), size)


def vertical_gradient(top, bottom):
    t = np.linspace(0.0, 1.0, H)[:, None, None]
    top = np.array(top, dtype=float)[None, None, :]
    bottom = np.array(bottom, dtype=float)[None, None, :]
    return np.repeat(top * (1 - t) + bottom * t, W, axis=1)


def main():
    img = vertical_gradient((5, 9, 17), (11, 22, 38))

    ys, xs = np.mgrid[0:H, 0:W].astype(float)

    # faint blueprint grid
    grid = ((xs % 48) < 1) | ((ys % 48) < 1)
    img[grid] += 9.0

    # lidar: center of the sweep on the right
    cx, cy = 1850.0, 350.0
    dx, dy = xs - cx, ys - cy
    r = np.hypot(dx, dy)
    theta = np.degrees(np.arctan2(dy, dx))  # -180..180

    # soft sector glow: smooth in both radius and angle, no hard edges
    def smoothstep(edge0, edge1, value):
        t = np.clip((value - edge0) / (edge1 - edge0), 0, 1)
        return t * t * (3 - 2 * t)

    radial = np.clip(1.0 - r / 440.0, 0, 1) ** 2
    angular = smoothstep(-158, -132, theta) * (1.0 - smoothstep(-58, -34, theta))
    glow_mask = Image.fromarray((radial * angular * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(18))
    glow = np.asarray(glow_mask, dtype=float) / 255.0 * 0.26
    img += glow[..., None] * np.array([10, 74, 96], dtype=float)

    # concentric range rings
    for radius in range(60, 331, 30):
        ring = np.abs(r - radius) < 1.1
        fade = max(0.05, 0.30 * (1.0 - radius / 390.0))
        img += ring[..., None] * CYAN * fade * 0.55

    # radial spokes
    for angle in (-118, -92, -66, -38):
        spoke = (np.abs(theta - angle) < 0.10) & (r > 70) & (r < 320)
        img += spoke[..., None] * CYAN * 0.16

    # ceiling: dot lattice suggesting voxels, fading to the right
    lattice = ((xs % 26) < 2) & ((ys % 26) < 2)
    fade_x = np.clip((xs - 1180) / 900.0, 0, 1) * np.clip(1 - (xs - 1180) / 1250.0, 0, 1)
    img += (lattice & (xs > 1180))[..., None] * CYAN * (fade_x[..., None] * 0.20)

    canvas = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    # scan returns: two noisy wall arcs plus sparse near returns
    rng = np.random.default_rng(20261110)
    pts = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pd = ImageDraw.Draw(pts)
    for radius, count, jitter, alpha in ((300, 300, 6.0, 150), (258, 90, 24.0, 85)):
        angles = rng.uniform(-124, -26, count)
        radii = radius + rng.normal(0, jitter, count)
        for a, rr in zip(angles, radii):
            x = cx + rr * np.cos(np.radians(a))
            y = cy + rr * np.sin(np.radians(a))
            s = float(rng.uniform(1.1, 2.6))
            c = tuple(int(v) for v in np.clip(WHITE * 0.55 + CYAN * 0.45, 0, 255))
            pd.ellipse((x - s, y - s, x + s, y + s), fill=c + (int(alpha * rng.uniform(0.5, 1.0)),))

    # origin crosshair
    pd.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(34, 211, 238, 230))
    pd.line((cx - 18, cy, cx - 9, cy), fill=(34, 211, 238, 120), width=2)
    pd.line((cx + 9, cy, cx + 18, cy), fill=(34, 211, 238, 120), width=2)
    pd.line((cx, cy - 18, cx, cy - 9), fill=(34, 211, 238, 120), width=2)
    pd.line((cx, cy + 9, cx, cy + 18), fill=(34, 211, 238, 120), width=2)

    # dashed planned trajectory leaving the sweep
    traj = [(1612, 596), (1668, 556), (1738, 570), (1798, 500), (1826, 436)]
    for (x0, y0), (x1, y1) in zip(traj, traj[1:]):
        steps = max(2, int(np.hypot(x1 - x0, y1 - y0) // 14))
        for i in range(steps):
            if i % 2:
                continue
            a = i / steps
            b = (i + 1) / steps
            pd.line((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a,
                     x0 + (x1 - x0) * b, y0 + (y1 - y0) * b),
                    fill=(245, 158, 11, 95), width=2)
    for x, y in traj[:-1]:
        pd.ellipse((x - 3.0, y - 3.0, x + 3.0, y + 3.0), outline=(245, 158, 11, 170), width=2)

    canvas = Image.alpha_composite(canvas, pts.filter(ImageFilter.GaussianBlur(0.4)))

    # gentle vignette keeps focus on the copy and the sweep
    vx = (xs - W / 2) / (W / 2)
    vy = (ys - H / 2) / (H / 2)
    vig = 1.0 - 0.22 * np.clip(vx ** 2 + vy ** 2, 0, 1) ** 1.4
    arr = np.asarray(canvas.convert("RGB"), dtype=float) * vig[..., None]
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    d = ImageDraw.Draw(canvas)

    # left edge accent
    d.rectangle((96, 142, 104, 540), fill=(34, 211, 238, 190))

    left = 140
    d.text((left, 136), "ROBOTAC  //  AUTONOMOUS  MAPPING", font=font(MONO, 27),
           fill=(56, 189, 248, 255))
    d.text((left - 4, 168), "Autonomous", font=font(SANS_B, 96), fill=(248, 250, 252, 255))
    d.text((left - 4, 262), "Mapping", font=font(SANS_B, 96), fill=(34, 211, 238, 255))
    d.text((left, 414), "UeSim + MuJoCo + ROS 2 Humble   ·   odometry-anchored voxel fusion",
           font=font(SANS, 29), fill=(148, 163, 184, 255))

    # metrics row
    metrics = [("30", "MIN BUDGET"), ("0.45", "M/S LIMIT"), ("2.5", "CM VOXEL"), ("12", "QA CHECKS")]
    x = left
    mono_small = font(MONO, 19)
    mono_val = font(MONO_B, 30)
    for value, label in metrics:
        d.text((x, 486), value, font=mono_val, fill=(248, 250, 252, 255))
        wv = d.textlength(value, font=mono_val)
        d.text((x + wv + 10, 496), label, font=mono_small, fill=(100, 116, 139, 255))
        x += wv + 10 + d.textlength(label, font=mono_small) + 42

    canvas.convert("RGB").save(OUT, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
