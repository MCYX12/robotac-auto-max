import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))
from repeat_runs import check_result, compare


def make_cloud(folder, shift):
    folder.mkdir()
    yy, zz = np.meshgrid(np.linspace(0, 5, 100), np.linspace(.3, 2.7, 40))
    points = np.column_stack([np.full(yy.size, shift), yy.ravel(), zz.ravel()]).astype('<f4')
    header = ('VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
              f'WIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA binary\n')
    path = folder / (folder.name + '_native.pcd')
    path.write_bytes(header.encode() + points.tobytes())
    return path


def test_repeatability_detects_displaced_surface_in_native_coordinates(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    make_cloud(first, 0)
    make_cloud(second, .5)
    result = compare(first, second)
    assert result['alignment'] == 'none'
    assert not result['pass']
    assert all(d['p95_m'] == pytest.approx(.5) for d in result['directions'])


def test_repeatability_accepts_unchanged_measurements(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    make_cloud(first, 0)
    make_cloud(second, 0)
    assert compare(first, second)['pass']


def test_success_label_without_version_or_geometry_is_rejected(tmp_path):
    folder = tmp_path / 'run'
    path = make_cloud(folder, 0)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {'state': 'COMPLETED', 'sha256': digest}
    (folder / 'result.json').write_text(json.dumps(record))
    assert not check_result(folder, 0)['pass']
    (folder / 'provenance.json').write_text(json.dumps({'software_sha256': {'a': 'b'}}))
    (folder / 'provenance_check.json').write_text(json.dumps({'pass': True}))
    (folder / 'geometry_audit.json').write_text(json.dumps({'proxy_check_pass': True, 'proxy_checks': {'test': False}}))
    assert not check_result(folder, 0)['pass']
    geometry = {'proxy_check_pass': True, 'proxy_checks': {'test': True}, 'native_sha256': digest,
                'reference_sha256': 'a'*64, 'auditor_sha256': 'b'*64, 'pcd_reader_sha256': 'c'*64,
                'audit_policy': {'test': 1}}
    record.update(audit_sources_sha256={'reference': 'a'*64, 'auditor': 'b'*64, 'reader': 'c'*64}, audit_policy=geometry['audit_policy'])
    (folder / 'result.json').write_text(json.dumps(record))
    (folder / 'geometry_audit.json').write_text(json.dumps(geometry))
    assert check_result(folder, 0)['pass']
    geometry['native_sha256'] = 'e'*64
    (folder / 'geometry_audit.json').write_text(json.dumps(geometry))
    assert not check_result(folder, 0)['pass']
    geometry['native_sha256'] = digest
    (folder / 'geometry_audit.json').write_text(json.dumps(geometry))
    path.write_bytes(path.read_bytes() + b'changed')
    assert not check_result(folder, 0)['pass']
