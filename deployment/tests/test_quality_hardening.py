"""Regression contracts for malformed clouds, hidden spatial faults and audit races."""
import fcntl
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deployment'))
sys.path.insert(0, str(ROOT/'dev_ws/tools'))
from pcd_quality_report import load_binary_pcd
import audit_run

spec = importlib.util.spec_from_file_location('quality_map_audit', ROOT/'deployment/validation/map_audit.py')
map_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(map_audit)


def write_pcd(path, points):
    data = np.asarray(points, dtype='<f4')
    header = f'VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(data)}\nHEIGHT 1\nPOINTS {len(data)}\nDATA binary\n'
    path.write_bytes(header.encode()+data.tobytes())
    return path


def test_binary_reader_accepts_valid_exact_layout(tmp_path):
    expected = np.array([[1.,2.,3.],[4.,5.,6.]])
    got, meta = load_binary_pcd(write_pcd(tmp_path/'native.pcd', expected))
    assert np.array_equal(got, expected) and meta['POINTS'] == ['2']


@pytest.mark.parametrize('before,after', [
    (b'WIDTH 2', b'WIDTH 3'),
    (b'POINTS 2', b'POINTS -2'),
    (b'HEIGHT 1', b'HEIGHT 1 2'),
    (b'COUNT 1 1 1', b'COUNT 1 1'),
    (b'SIZE 4 4 4', b'SIZE 4 4'),
    (b'TYPE F F F', b'TYPE F F Q'),
    (b'FIELDS x y z', b'FIELDS x x z'),
    (b'COUNT 1 1 1', b'COUNT 2 1 1'),
    (b'DATA binary', b'POINTS 2\nDATA binary'),
    (b'WIDTH 2', b'WIDTH 100000001'),
])
def test_binary_reader_rejects_inconsistent_headers(tmp_path, before, after):
    path = write_pcd(tmp_path/'bad.pcd', [[1,2,3],[4,5,6]])
    path.write_bytes(path.read_bytes().replace(before,after))
    with pytest.raises(ValueError):load_binary_pcd(path)


@pytest.mark.parametrize('extra', [b'x', b'\0'*12])
def test_binary_reader_rejects_undeclared_payload(tmp_path, extra):
    path=write_pcd(tmp_path/'bad.pcd',[[1,2,3],[4,5,6]])
    path.write_bytes(path.read_bytes()+extra)
    with pytest.raises(ValueError, match='payload length'):load_binary_pcd(path)


def test_binary_reader_rejects_truncated_payload(tmp_path):
    path=write_pcd(tmp_path/'bad.pcd',[[1,2,3],[4,5,6]])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='payload length'):load_binary_pcd(path)


def box_scene(tmp_path):
    xml=tmp_path/'reference.xml'
    xml.write_text('<mujoco><worldbody>'+
      '<geom name="north" type="box" pos="0 2.1 1.5" size="2.2 .1 1.5"/>'+
      '<geom name="south" type="box" pos="0 -2.1 1.5" size="2.2 .1 1.5"/>'+
      '<geom name="east" type="box" pos="2.1 0 1.5" size="2.2 .1 1.5" quat=".707106781187 0 0 .707106781187"/>'+
      '<geom name="west" type="box" pos="-2.1 0 1.5" size="2.2 .1 1.5" quat=".707106781187 0 0 .707106781187"/>'+
      '</worldbody></mujoco>')
    v=np.arange(-1.975,2,.05);x,y=np.meshgrid(v,v)
    cloud=[np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])]
    along,z=np.meshgrid(np.arange(-2,2.001,.05),np.arange(.05,3.001,.05))
    for side in (-2,2):
        cloud.append(np.column_stack([along.ravel(),np.full(along.size,side),z.ravel()]))
        cloud.append(np.column_stack([np.full(along.size,side),along.ravel(),z.ravel()]))
    return xml,np.vstack(cloud)


def test_high_outside_scene_patch_cannot_hide_from_geometry_gate(tmp_path):
    xml,cloud=box_scene(tmp_path)
    x,y=np.meshgrid(np.arange(100,102,.05),np.arange(100,102,.05))
    injected=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,3.1)])
    pcd=write_pcd(tmp_path/'injected.pcd',np.vstack([cloud,injected]))
    result=map_audit.audit(pcd,xml,tmp_path/'audit.json')
    original_names=['floor_coverage_at_10cm_ge_99_5pct','floor_coverage_at_20cm_ge_99_999pct',
        'largest_floor_gap_radius_le_25cm','wall_coverage_at_15cm_ge_99_5pct',
        'wall_median_reference_residual_le_5mm','wall_outliers_beyond_20cm_le_0_01pct']
    assert all(result['proxy_checks'][name] for name in original_names)
    assert not result['proxy_check_pass']
    assert not result['proxy_checks']['all_height_footprint_outliers_beyond_35cm_le_0_01pct']
    assert result['all_point_spatial_support']['outside_footprint_points']==len(injected)
    assert result['height_band_support']['upper_wall']['outside_footprint_points']==len(injected)
    assert result['all_point_spatial_support']['clusters'][0]['points']==len(injected)


def test_local_wall_gap_is_not_diluted_by_larger_walls(tmp_path):
    xml,cloud=box_scene(tmp_path)
    # A 60 cm square hole in one wall must fail its local maximum-gap gate.
    remove=(cloud[:,1]>1.99)&(cloud[:,1]<2.01)&(np.abs(cloud[:,0])<.35)&(cloud[:,2]>.8)&(cloud[:,2]<1.5)
    pcd=write_pcd(tmp_path/'hole.pcd',cloud[~remove])
    result=map_audit.audit(pcd,xml,tmp_path/'audit.json')
    assert not result['proxy_checks']['each_wall_largest_gap_le_25cm']
    assert any(w['max_missing_distance_m']>.25 for w in result['worst_walls'])
    assert set(result['additional_surface_diagnostics']) >= {'wall_lower_strip','wall_upper_strip'}


@pytest.fixture
def audit_fixture(tmp_path,monkeypatch):
    root=tmp_path/'project';folder=root/'dev_ws/midterm/runs/fixture';folder.mkdir(parents=True)
    pcd=folder/'fixture_native.pcd';pcd.write_bytes(b'fixture native bytes')
    digest=hashlib.sha256(pcd.read_bytes()).hexdigest()
    result={'state':'VALIDATING','run_id':'fixture','supervisor_pid':1234,'supervisor_start_ticks':5678,'sha256':digest}
    (folder/'result.json').write_text(json.dumps(result));(folder/'fixture_motion.json').write_text(json.dumps({'state':'MAP_SAVED'}))
    paths={'reference':root/'runtime/src/robot_mujoco/zsibot_robots/xgb/unreal.xml',
        'auditor':root/'deployment/validation/map_audit.py','reader':root/'dev_ws/tools/pcd_quality_report.py'}
    for key,path in paths.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_text(key)
    (root/'deployment/run_status.json').write_text(json.dumps(result))
    monkeypatch.setattr(audit_run,'ROOT',root);monkeypatch.setattr(audit_run,'STOP',root/'deployment/stop.request')
    report={'proxy_checks':{name:True for name in audit_run.REQUIRED_CHECKS},'proxy_check_pass':True,'quality_state':'PASS_PROXY_CHECKS',
        'native_sha256':digest,'reference_sha256':audit_run.sha256(paths['reference']),
        'auditor_sha256':audit_run.sha256(paths['auditor']),'pcd_reader_sha256':audit_run.sha256(paths['reader']),
        'limitations':['synthetic fixture'],'audit_policy':{'version':'fixture'},
        'floor_coverage':{'0.1':1.,'0.2':1.},'interior_wall_surface_coverage':{'0.15':1.},
        'floor_largest_gap_radius_m':.01,'wall_point_fraction_beyond':{'0.2':0.},
        'worst_walls':[{'surface_coverage_20cm':1.,'max_missing_distance_m':.01}],
        'additional_surface_diagnostics':{'wall_upper_strip':{'coverage_at_20cm':1.,'maximum_gap_m':.01}},
        'all_point_spatial_support':{'outside_footprint_points':0,'outside_footprint_fraction':0.}}
    def finish(*args,**kwargs):
        (folder/'geometry_audit.json').write_text(json.dumps(report))
    monkeypatch.setattr(audit_run.subprocess,'run',finish)
    return root,folder,pcd,result,report,finish


def test_manual_audit_cannot_take_over_active_validation(audit_fixture):
    _,folder,_,_,_,_=audit_fixture
    with pytest.raises(ValueError,match='belongs to the run supervisor'):audit_run.audit_run(folder)


@pytest.mark.parametrize('terminal',['FAILED','CANCELLED'])
def test_cancellation_or_failure_during_audit_cannot_be_promoted(audit_fixture,monkeypatch,terminal):
    _,folder,_,initial,_,finish=audit_fixture
    def changed(*args,**kwargs):
        finish(*args,**kwargs)
        (folder/'result.json').write_text(json.dumps(dict(initial,state=terminal)))
    monkeypatch.setattr(audit_run.subprocess,'run',changed)
    with pytest.raises(ValueError,match='failed or was cancelled'):audit_run.audit_run(folder,1234,5678)
    assert json.loads((folder/'result.json').read_text())['state']==terminal


def test_motion_cancellation_during_audit_cannot_be_promoted(audit_fixture,monkeypatch):
    _,folder,_,_,_,finish=audit_fixture
    def changed(*args,**kwargs):
        finish(*args,**kwargs);(folder/'fixture_motion.json').write_text('{"state":"CANCELLED"}')
    monkeypatch.setattr(audit_run.subprocess,'run',changed)
    with pytest.raises(ValueError,match='failed or was cancelled'):audit_run.audit_run(folder,1234,5678)


def test_owner_change_during_audit_cannot_be_promoted(audit_fixture,monkeypatch):
    _,folder,_,initial,_,finish=audit_fixture
    def changed(*args,**kwargs):
        finish(*args,**kwargs);(folder/'result.json').write_text(json.dumps(dict(initial,supervisor_start_ticks=999)))
    monkeypatch.setattr(audit_run.subprocess,'run',changed)
    with pytest.raises(ValueError,match='ownership'):audit_run.audit_run(folder,1234,5678)


def test_native_replacement_during_audit_cannot_be_promoted(audit_fixture,monkeypatch):
    _,folder,pcd,_,_,finish=audit_fixture
    def changed(*args,**kwargs):finish(*args,**kwargs);pcd.write_bytes(b'replaced')
    monkeypatch.setattr(audit_run.subprocess,'run',changed)
    with pytest.raises(ValueError,match='changed during validation'):audit_run.audit_run(folder,1234,5678)


def test_audit_requires_unchanged_source_provenance(audit_fixture):
    _,folder,_,_,report,_=audit_fixture;report['reference_sha256']='0'*64
    with pytest.raises(ValueError,match='provenance'):audit_run.audit_run(folder,1234,5678)


def test_two_audits_cannot_write_same_result(audit_fixture):
    _,folder,_,_,_,_=audit_fixture
    with (folder/'.audit.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='already in progress'):audit_run.audit_run(folder,1234,5678)


def test_healthy_owned_audit_commits_consistent_result(audit_fixture):
    root,folder,_,_,_,_=audit_fixture
    assert audit_run.audit_run(folder,1234,5678)==0
    result=json.loads((folder/'result.json').read_text())
    assert result['state']=='COMPLETED' and not result['perfect_map_certified']
    assert json.loads((root/'deployment/run_status.json').read_text())==result
    assert json.loads((folder/'fixture_motion.json').read_text())['state']=='COMPLETED'


def test_stop_request_during_audit_cannot_be_promoted(audit_fixture,monkeypatch):
    root,folder,_,_,_,finish=audit_fixture
    def changed(*args,**kwargs):
        finish(*args,**kwargs);(root/'deployment/stop.request').touch()
    monkeypatch.setattr(audit_run.subprocess,'run',changed)
    with pytest.raises(ValueError,match='stopped or was cancelled'):audit_run.audit_run(folder,1234,5678)


def test_stale_or_partial_check_schema_is_rejected(audit_fixture):
    _,folder,_,_,report,_=audit_fixture
    del report['proxy_checks']['each_wall_largest_gap_le_25cm']
    with pytest.raises(ValueError,match='invalid checks'):audit_run.audit_run(folder,1234,5678)


def test_contradictory_verdict_is_rejected(audit_fixture):
    _,folder,_,_,report,_=audit_fixture
    report['proxy_checks']['each_wall_largest_gap_le_25cm']=False
    with pytest.raises(ValueError,match='does not match'):audit_run.audit_run(folder,1234,5678)


def test_missing_top_band_cannot_pass_only_the_legacy_main_wall_band(tmp_path):
    xml,cloud=box_scene(tmp_path)
    # Retain the legacy targets through 2.7 m but omit the roof-adjacent strip.
    cloud=cloud[cloud[:,2]<=2.70001]
    result=map_audit.audit(write_pcd(tmp_path/'top_missing.pcd',cloud),xml,tmp_path/'audit.json')
    assert result['proxy_checks']['wall_coverage_at_15cm_ge_99_5pct']
    assert not result['proxy_checks']['top_strip_coverage_at_20cm_ge_99_5pct']
    assert not result['proxy_check_pass']


def test_reference_start_inside_wall_is_rejected():
    with pytest.raises(ValueError,match='inside a collision box'):
        map_audit.scene_grid([('blocked',np.array([0.,0.,1.5]),np.array([1.,1.,1.5]),np.eye(2))])
