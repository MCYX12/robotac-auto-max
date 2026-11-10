#!/usr/bin/env python3
"""Publish a validated XYZ PCD as a transient-local ROS PointCloud2 map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String


_PCD_TYPES = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("U", 1): "u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_task1_evidence(path: Path, digest: str, finite_points: int) -> dict:
    """Require the native Task-1 completion evidence stored beside the PCD."""
    status_path = path.parent / "task1_status.json"
    validation_path = path.parent / "artifact_validation.json"
    checksums_path = path.parent / "SHA256SUMS.txt"
    for evidence in (status_path, validation_path, checksums_path):
        if not evidence.is_file():
            raise ValueError(f"missing task-one provenance evidence: {evidence}")
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid task-one provenance evidence: {error}") from error
    if status.get("state") != "COMPLETED" or status.get("reason") != (
        "native_pcd_saved_and_validated"
    ):
        raise ValueError("task-one status does not prove native PCD completion")
    if Path(str(status.get("output_pcd", ""))).name != path.name:
        raise ValueError("task-one status names a different PCD")
    if validation.get("state") != "PASS":
        raise ValueError("task-one artifact validation did not pass")
    if int(validation.get("points", 0)) != finite_points:
        raise ValueError("task-one artifact point count differs from the loaded PCD")
    checksum_records = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            checksum_records[parts[1].lstrip("*./")] = parts[0].lower()
    if checksum_records.get(path.name) != digest.lower():
        raise ValueError("task-one PCD SHA-256 differs from its sealed checksum")
    return {
        "task1_status": str(status_path),
        "artifact_validation": str(validation_path),
        "sealed_checksums": str(checksums_path),
    }


def load_xyz_pcd(path: Path, maximum_points: int = 0) -> tuple[np.ndarray, dict]:
    """Load finite XYZ values from an ASCII or uncompressed-binary PCD."""
    header: dict[str, list[str]] = {}
    header_bytes = 0
    header_lines = 0
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PCD header ended before DATA")
            header_bytes += len(raw)
            header_lines += 1
            line = raw.decode("ascii", errors="strict").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break

    fields = header.get("FIELDS", [])
    sizes = [int(value) for value in header.get("SIZE", [])]
    types = [value.upper() for value in header.get("TYPE", [])]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    if not fields or not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("inconsistent PCD FIELDS/SIZE/TYPE/COUNT")
    if any(count != 1 for count in counts):
        raise ValueError("only scalar PCD fields are supported")
    for required in ("x", "y", "z"):
        if required not in fields:
            raise ValueError(f"PCD is missing {required} field")
    points = int(header.get("POINTS", ["0"])[0])
    if points <= 0:
        width = int(header.get("WIDTH", ["0"])[0])
        height = int(header.get("HEIGHT", ["1"])[0])
        points = width * height
    if points <= 0:
        raise ValueError("PCD point count must be positive")

    mode = header["DATA"][0].lower()
    if mode == "binary":
        formats = []
        for kind, size in zip(types, sizes):
            try:
                formats.append(_PCD_TYPES[(kind, size)])
            except KeyError as error:
                raise ValueError(f"unsupported PCD field type {kind}{size}") from error
        dtype = np.dtype({"names": fields, "formats": formats})
        expected = points * dtype.itemsize
        with path.open("rb") as stream:
            stream.seek(header_bytes)
            payload = stream.read(expected)
        if len(payload) != expected:
            raise ValueError(
                f"truncated PCD payload: expected {expected} bytes, got {len(payload)}"
            )
        records = np.frombuffer(payload, dtype=dtype, count=points)
        xyz = np.column_stack((records["x"], records["y"], records["z"]))
    elif mode == "ascii":
        values = np.loadtxt(path, dtype=np.float64, skiprows=header_lines, ndmin=2)
        if values.shape[0] != points or values.shape[1] != len(fields):
            raise ValueError("ASCII PCD dimensions do not match its header")
        xyz = values[:, [fields.index("x"), fields.index("y"), fields.index("z")]]
    else:
        raise ValueError(f"unsupported PCD DATA mode: {mode}")

    xyz = np.asarray(xyz, dtype=np.float32)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if not len(xyz):
        raise ValueError("PCD has no finite XYZ points")
    source_points = int(len(xyz))
    if maximum_points > 0 and len(xyz) > maximum_points:
        indices = np.linspace(0, len(xyz) - 1, maximum_points, dtype=np.int64)
        xyz = xyz[indices]
    metadata = {
        "data_mode": mode,
        "declared_points": points,
        "finite_points": source_points,
        "published_points": int(len(xyz)),
    }
    return np.ascontiguousarray(xyz), metadata


def xyz_message(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    message = PointCloud2()
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    message.height = 1
    message.width = int(len(points))
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = message.point_step * message.width
    message.is_dense = True
    message.data = points.astype("<f4", copy=False).tobytes()
    return message


class PcdMapServer(Node):
    def __init__(self) -> None:
        super().__init__("robotac_pcd_map_server")
        self.declare_parameter("pcd_path", "")
        self.declare_parameter("cloud_topic", "/robotac/task2/known_cloud")
        self.declare_parameter("status_topic", "/robotac/task2/map_status")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("maximum_points", 1500000)

        self.path = Path(str(self.get_parameter("pcd_path").value)).expanduser()
        self.cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.maximum_points = int(self.get_parameter("maximum_points").value)
        if not self.path.is_file():
            raise FileNotFoundError(f"known-map PCD does not exist: {self.path}")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.cloud_publisher = self.create_publisher(PointCloud2, self.cloud_topic, qos)
        self.status_publisher = self.create_publisher(String, self.status_topic, qos)
        self.points, self.metadata = load_xyz_pcd(self.path, self.maximum_points)
        digest = sha256_file(self.path)
        evidence = validate_task1_evidence(
            self.path, digest, int(self.metadata["finite_points"])
        )
        self.metadata.update(
            {
                "state": "READY",
                "source_kind": "task1_native_pcd",
                "path": str(self.path),
                "sha256": digest,
                "frame_id": self.frame_id,
                "provenance_evidence": evidence,
            }
        )
        self._published = False
        self.create_timer(0.25, self._publish_once)

    def _publish_once(self) -> None:
        if self._published:
            return
        stamp = self.get_clock().now().to_msg()
        self.cloud_publisher.publish(xyz_message(self.points, self.frame_id, stamp))
        self.status_publisher.publish(
            String(data=json.dumps(self.metadata, separators=(",", ":")))
        )
        self._published = True
        self.get_logger().info(json.dumps(self.metadata, ensure_ascii=False))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PcdMapServer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
