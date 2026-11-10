"""Fingerprint the listed local software and calibration, without secrets."""
import hashlib
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, atomic_json, read_json


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def runtime_files(root=ROOT):
    root = Path(root)
    files = set()
    for folder, suffixes in (
        ('deployment', {'.py'}),
        ('dev_ws/tools', {'.py'}),
        ('dev_ws/src', {'.cpp', '.hpp', '.h', '.yaml', '.xml', '.txt'}),
        ('finals_ws/src/robotac_finals', {'.py', '.yaml', '.xml', '.cfg', '.rviz'}),
        ('finals_ws/test', {'.py'}),
        ('deps/zsibot_sdk/lib/zsl-1/x86_64', {'.so', '.py'}),
    ):
        for path in (root / folder).rglob('*'):
            if path.is_file() and path.suffix in suffixes and '__pycache__' not in path.parts and 'tests' not in path.parts:
                files.add(str(path.relative_to(root)))
    for name in (
        'docker.sh', 'run_auto_max.sh', 'build.sh', 'deployment/env.sh',
        'dev_ws/midterm/autonomous.launch.py',
        'config/config.json', 'runtime/config/config.json',
        'runtime/src/UeSim/Linux/UeSim/Content/model/config/config.json',
        'runtime/src/robot_mc/build/export/config/sdk_config.yaml',
        'runtime/src/robot_mujoco/zsibot_robots/xgb/unreal.xml',
        'deployment/SHA256SUMS.runtime.txt',
        'dev_ws/install_humble/robotac_mapping_baseline/lib/robotac_mapping_baseline/odom_voxel_mapper',
    ):
        if (root / name).is_file():
            files.add(name)
    files.update(vendor_manifest(root))
    return sorted(files)


def vendor_manifest(root):
    path = Path(root) / 'deployment/SHA256SUMS.runtime.txt'
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        checksum, name = line.split('  ', 1)
        relative = Path(name)
        if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum) or relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('invalid_vendor_checksum_manifest')
        result[name] = checksum
    return result


def capture(folder, settings, root=ROOT):
    """Called after preflight, before launch. Writes only the new run directory."""
    root = Path(root)
    hashes = {name: digest(root / name) for name in runtime_files(root)}
    if not hashes:
        raise RuntimeError('provenance_has_no_runtime_files')
    if any(hashes.get(name) != expected for name, expected in vendor_manifest(root).items()):
        raise RuntimeError('vendor_executable_checksum_mismatch')
    state = read_json(root / 'deployment/runtime_status.json')
    payload = {
        'schema_version': 1,
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'settings': dict(settings),
        'software_sha256': hashes,
        'runtime_generation': state.get('generation'),
        'runtime_log_directory': state.get('log_directory'),
        'environment': {key: os.environ.get(key) for key in (
            'ROS_DISTRO', 'ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION',
            'ROS_LOCALHOST_ONLY', 'ROBOTAC_HEADLESS', 'MIDTERM_RVIZ',
            'ROBOTAC_STATIONARY_MAPPING',
        )},
        'python_version': sys.version,
        'machine': platform.machine(),
        'kernel': platform.release(),
        'calibration_note': 'Hashes include both sensor configurations, SDK config and the offline-only reference. The reference is not an input to navigation or mapping.',
    }
    atomic_json(Path(folder) / 'provenance.json', payload)
    return payload


def verify(folder, root=ROOT):
    """Fail if runtime source/calibration changed during an acquisition."""
    root = Path(root)
    record = read_json(Path(folder) / 'provenance.json')
    expected = record.get('software_sha256')
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError('missing_run_provenance')
    changed = []
    for name, before in expected.items():
        # Reject a modified record asking us to hash files outside this project.
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('invalid_provenance_path')
        path = root / relative
        if not path.is_file() or digest(path) != before:
            changed.append(name)
    added = sorted(set(runtime_files(root)) - set(expected))
    changed.extend(added)
    result = {'checked_files': len(expected), 'changed_files': sorted(set(changed)), 'pass': not changed}
    atomic_json(Path(folder) / 'provenance_check.json', result)
    if changed:
        raise RuntimeError('runtime_files_changed_during_run:' + ','.join(changed[:8]))
    return result
