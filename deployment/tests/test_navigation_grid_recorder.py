import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import pytest
for folder in (Path(__file__).parent,Path(__file__).parents[1]/'validation'):
    if (folder/'record_navigation_grid.py').exists():sys.path.insert(0,str(folder));break
import record_navigation_grid as recorder


def message():
    from nav_msgs.msg import OccupancyGrid
    m=OccupancyGrid();m.header.stamp.sec=10;m.header.frame_id='map'
    m.info.width=3;m.info.height=2;m.info.resolution=.1;m.info.origin.orientation.w=1.
    m.info.origin.position.x=-2.;m.info.map_load_time.sec=1;m.data=[-1,0,25,49,50,100]
    return m


def bare_recorder(tmp_path):
    node=object.__new__(recorder.GridRecorder);runtime=tmp_path/'runtime.json';run=tmp_path/'run.json'
    recorder.atomic_json(runtime,{'state':'RUNNING','generation':'g'});recorder.atomic_json(run,{'state':'RUNNING','run_id':'r'})
    node.args=SimpleNamespace(output=tmp_path,interval=60.,runtime_status=runtime,run_status=run,run_id='r')
    node.generation='g';node.started=time.monotonic();node.started_wall_ns=time.time_ns();node.stats=defaultdict(int)
    node.latest=None;node.last_saved_monotonic=node.started-60;node.parts=[];node.failure=None;node.close_requested=None;node.missing_state_since=None
    node.owner={'pid':123,'run_id':'r','generation':'g'};node.source={'script_sha256':recorder.sha(recorder.__file__),'run_id':'r','runtime_generation':'g'}
    return node


def test_exact_grid_values_layout_and_metadata_are_preserved():
    m=message();data,meta=recorder.parse_grid(m)
    assert data.dtype==np.int8 and data.tolist()==[[-1,0,25],[49,50,100]]
    assert meta['origin_position_m']==[-2.,0.,0.] and meta['origin_quaternion_xyzw']==[0.,0.,0.,1.]
    assert meta['unknown_cells']==1 and meta['free_cells']==1 and meta['occupied_at_50_cells']==2
    m.data[0]=100
    assert data[0,0]==-1


@pytest.mark.parametrize('bad',['size','nan_resolution','zero_resolution','nan_origin','zero_quaternion','negative_stamp','empty_frame','bad_occupancy'])
def test_invalid_grid_rejected(bad):
    m=message()
    if bad=='size':m.info.width=4
    elif bad=='nan_resolution':m.info.resolution=float('nan')
    elif bad=='zero_resolution':m.info.resolution=0.
    elif bad=='nan_origin':m.info.origin.position.z=float('nan')
    elif bad=='zero_quaternion':m.info.origin.orientation.w=0.
    elif bad=='negative_stamp':m.header.stamp.sec=-1
    elif bad=='empty_frame':m.header.frame_id=''
    else:m.data[1]=-2
    with pytest.raises(ValueError):recorder.parse_grid(m)


@pytest.mark.parametrize('runtime,run,expected',[
    ({'state':'RUNNING','generation':'new'},{'state':'RUNNING','run_id':'r'},('runtime_generation_changed',None)),
    ({'state':'RUNNING','generation':'g'},{'state':'RUNNING','run_id':'other'},('run_id_changed',None)),
    ({'state':'RUNNING','generation':'g'},{'state':'SAVING','run_id':'r'},(None,'acquisition_finished')),
    ({'state':'STOPPED','generation':'g'},{'state':'RUNNING','run_id':'r'},(None,'runtime_stopped')),
    ({'state':'STOPPED','generation':'g'},{'state':'CANCELLED','run_id':'r','reason':'test'},('run_cancelled:test',None)),
    ({'state':'FAILED','generation':'g','reason':'test'},{'state':'RUNNING','run_id':'r'},('runtime_failed:test',None)),
])
def test_lifecycle_never_mixes_runs(runtime,run,expected):
    assert recorder.lifecycle(runtime,run,'r','g')==expected


def test_snapshot_hashes_final_latest_and_same_stamp_update(tmp_path):
    node=bare_recorder(tmp_path);m=message();node.on_grid(m);node.monitor()
    assert len(node.parts)==1
    m.data[0]=100;node.on_grid(m)
    assert node.close_capture('acquisition_finished')==0
    manifest=json.loads((tmp_path/'capture_manifest.json').read_text())
    assert manifest['capture_state']=='CLOSED' and manifest['final_latest_stamp_saved'] and len(manifest['snapshots'])==2
    for entry in manifest['snapshots']:
        p=tmp_path/entry['name'];assert recorder.sha(p)==entry['sha256'];assert recorder.sha(p.with_suffix('.json'))==entry['metadata_sha256']
    with np.load(tmp_path/'grid_0001.npz',allow_pickle=False) as result:
        assert result['occupancy'][0,0]==100 and result['frame_id'].item()=='map'


def test_no_grid_cannot_close_successfully(tmp_path):
    node=bare_recorder(tmp_path)
    assert node.close_capture('acquisition_finished')==1
    assert json.loads((tmp_path/'capture_manifest.json').read_text())['capture_state']=='INCOMPLETE'


def test_timestamp_regression_stops_capture(tmp_path):
    node=bare_recorder(tmp_path);m=message();node.on_grid(m);m.header.stamp.sec=9;node.on_grid(m)
    assert node.failure=='navigation_grid_timestamp_regressed'
    assert node.close_capture('acquisition_finished')==1


def test_new_generation_is_rejected_before_grid_is_retained(tmp_path):
    node=bare_recorder(tmp_path);recorder.atomic_json(node.args.runtime_status,{'state':'RUNNING','generation':'new'})
    node.on_grid(message());assert node.latest is None and node.failure=='runtime_generation_changed'


def test_manual_end_is_incomplete_even_when_grid_was_saved(tmp_path):
    node=bare_recorder(tmp_path);node.on_grid(message())
    assert node.close_capture('operator_stopped')==1
    assert json.loads((tmp_path/'capture_manifest.json').read_text())['close_reason']=='operator_stopped'


def test_snapshot_write_failure_cannot_claim_closed(tmp_path,monkeypatch):
    node=bare_recorder(tmp_path);node.on_grid(message())
    def fail(_):raise OSError('injected disk failure')
    monkeypatch.setattr(node,'save_latest',fail)
    assert node.close_capture('acquisition_finished')==1
    assert 'snapshot_write_failed' in json.loads((tmp_path/'capture_manifest.json').read_text())['close_reason']
