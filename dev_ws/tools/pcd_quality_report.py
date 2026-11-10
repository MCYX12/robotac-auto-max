#!/usr/bin/env python3
"""Compute deterministic sanity/noise proxies for an uncompressed PCD map."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PCD_DTYPES = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("U", 1): "u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
}


def load_binary_pcd(path: Path):
    """Read a structurally valid, bounded, uncompressed PCD without guessing fields."""
    path = Path(path)
    metadata = {}
    with path.open("rb") as stream:
        header_bytes = 0
        while True:
            raw = stream.readline(65537)
            if not raw:
                raise ValueError("PCD header ended before DATA")
            header_bytes += len(raw)
            if header_bytes > 65536:
                raise ValueError("PCD header exceeds 64 KiB")
            line = raw.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            key = key.upper()
            if key in metadata:
                raise ValueError("duplicate PCD header field: " + key)
            metadata[key] = values
            if key.upper() == "DATA":
                if values != ["binary"]:
                    raise ValueError("only uncompressed binary PCD is supported")
                break
        for key in ("FIELDS", "SIZE", "TYPE", "WIDTH", "HEIGHT", "POINTS"):
            if key not in metadata:
                raise ValueError("missing PCD header field: " + key)
        fields = metadata["FIELDS"]
        if not 3 <= len(fields) <= 32 or len(set(fields)) != len(fields):
            raise ValueError("PCD fields must be unique and contain 3..32 entries")
        if not set("xyz").issubset(fields):
            raise ValueError("PCD must contain x, y and z fields")
        sizes = [int(v) for v in metadata["SIZE"]]
        types = metadata["TYPE"]
        counts = [int(v) for v in metadata.get("COUNT", ["1"] * len(fields))]
        if not len(fields) == len(sizes) == len(types) == len(counts):
            raise ValueError("PCD FIELDS/SIZE/TYPE/COUNT lengths differ")
        shape = {}
        for key in ("WIDTH", "HEIGHT", "POINTS"):
            if len(metadata[key]) != 1:
                raise ValueError("PCD " + key + " must contain one integer")
            shape[key] = int(metadata[key][0])
            if shape[key] <= 0 or shape[key] > 100_000_000:
                raise ValueError("PCD " + key + " is outside supported bounds")
        if shape["WIDTH"] * shape["HEIGHT"] != shape["POINTS"]:
            raise ValueError("PCD WIDTH * HEIGHT differs from POINTS")
        dtype_fields = []
        for name, size, kind, count in zip(fields, sizes, types, counts):
            if (kind, size) not in PCD_DTYPES or not 1 <= count <= 1024:
                raise ValueError("unsupported PCD field type/size/count")
            if name in ("x", "y", "z") and count != 1:
                raise ValueError("PCD XYZ fields must be scalar")
            scalar = PCD_DTYPES[(kind, size)]
            dtype_fields.append((name, scalar) if count == 1 else (name, scalar, (count,)))
        dtype = np.dtype(dtype_fields)
        expected_bytes = shape["POINTS"] * dtype.itemsize
        if expected_bytes > 2_000_000_000:
            raise ValueError("PCD payload exceeds supported 2 GB bound")
        if path.stat().st_size - stream.tell() != expected_bytes:
            raise ValueError("PCD payload length differs from declared point layout")
        payload = stream.read(expected_bytes)
        if len(payload) != expected_bytes or stream.read(1):
            raise ValueError("PCD changed during read")
    records = np.frombuffer(payload, dtype=dtype)
    xyz = np.column_stack([records[axis].astype(np.float64) for axis in "xyz"])
    return xyz, metadata


def report(path: Path):
    xyz, metadata = load_binary_pcd(path)
    finite_mask = np.isfinite(xyz).all(axis=1)
    points = xyz[finite_mask]
    if len(points) < 2:
        raise ValueError("map has fewer than two finite XYZ points")

    nearest = cKDTree(points).query(points, k=2, workers=-1)[0][:, 1]
    low = np.percentile(points, [1, 50, 99], axis=0)
    result = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pcd_structure_valid": True,
        "pcd_declared_points": int(metadata["POINTS"][0]),
        "decoded_points": int(len(xyz)),
        "finite_points": int(len(points)),
        "invalid_ratio": float(1.0 - finite_mask.mean()),
        "bbox_min_xyz_m": points.min(axis=0).tolist(),
        "bbox_max_xyz_m": points.max(axis=0).tolist(),
        "percentile_01_xyz_m": low[0].tolist(),
        "percentile_50_xyz_m": low[1].tolist(),
        "percentile_99_xyz_m": low[2].tolist(),
        "nearest_neighbor_m": {
            "median": float(np.median(nearest)),
            "p95": float(np.percentile(nearest, 95)),
            "p99": float(np.percentile(nearest, 99)),
            "isolated_over_0_10_ratio": float(np.mean(nearest > 0.10)),
            "isolated_over_0_20_ratio": float(np.mean(nearest > 0.20)),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcd", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [report(path) for path in args.pcd]
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
