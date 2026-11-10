from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import capture, verify


def test_changed_calibration_or_software_is_not_certified(tmp_path):
    root = tmp_path / 'project'
    config = root / 'config/config.json'
    config.parent.mkdir(parents=True)
    config.write_text('{"fov":120}')
    folder = root / 'run'
    capture(folder, {'duration_seconds': 1800}, root)
    assert verify(folder, root)['pass']
    config.write_text('{"fov":90}')
    with pytest.raises(RuntimeError, match='runtime_files_changed_during_run'):
        verify(folder, root)


def test_removed_or_added_runtime_source_is_detected(tmp_path):
    root = tmp_path / 'project'
    source = root / 'deployment/control.py'
    source.parent.mkdir(parents=True)
    source.write_text('version = 1\n')
    folder = root / 'run'
    capture(folder, {}, root)
    (source.parent / 'new_runtime.py').write_text('version = 2\n')
    with pytest.raises(RuntimeError, match='runtime_files_changed_during_run'):
        verify(folder, root)
    (source.parent / 'new_runtime.py').unlink()
    source.unlink()
    with pytest.raises(RuntimeError, match='runtime_files_changed_during_run'):
        verify(folder, root)


def test_missing_provenance_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match='missing_run_provenance'):
        verify(tmp_path, tmp_path)


def test_actual_vendor_binary_is_verified_not_only_manifest(tmp_path):
    import hashlib
    root = tmp_path / 'project'
    (root / 'deployment').mkdir(parents=True)
    binary = root / 'vendor.bin'
    binary.write_bytes(b'original')
    checksum = hashlib.sha256(binary.read_bytes()).hexdigest()
    (root / 'deployment/SHA256SUMS.runtime.txt').write_text(checksum + '  vendor.bin\n')
    folder = root / 'run'
    capture(folder, {}, root)
    binary.write_bytes(b'modified')
    with pytest.raises(RuntimeError, match='runtime_files_changed_during_run'):
        verify(folder, root)
    with pytest.raises(RuntimeError, match='vendor_executable_checksum_mismatch'):
        capture(root / 'another_run', {}, root)
