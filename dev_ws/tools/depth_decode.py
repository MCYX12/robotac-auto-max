"""Decode actual Matrix depth returns, independent of offline coverage diagnostics."""
import numpy as np


def decode_ground_candidates(image, stride=4):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError('Expected encoded BGR/BGRA uint8 PNG')
    depth = image[:, :, 2].astype(float) + image[:, :, 1] / 255.0
    h, w = depth.shape
    focal = w / (2 * np.tan(np.deg2rad(120) / 2))
    v, u = np.mgrid[4:h-4:stride, 4:w-4:stride]
    d = depth[v, u]
    valid = (d >= .6) & (d <= 4)
    for dv, du in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        valid &= np.abs(depth[v + dv, u + du] - d) < .05
    xyz = np.stack((d, -(u - w / 2) * d / focal, -(v - h / 2) * d / focal), axis=-1)
    return xyz[valid]
