import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest
for folder in (Path(__file__).parent,Path(__file__).parents[1]/'validation'):
    if (folder/'temporal_consistency.py').exists():sys.path.insert(0,str(folder));break
from temporal_consistency import analyse,DEFAULTS,interpolate_pose


def fixture(x=5.2,repeat=True):
    t=np.array([0,.5,1,1.5,80,80.5,81,81.5,160,160.5,161,161.5] if repeat else [0,.5,1,1.5,2,2.5])
    walls=[]
    for i in range(3):
        y,z=np.meshgrid(np.linspace(i+.1,i+.9,14),np.linspace(.5,2.5,18))
        walls.append(np.column_stack([np.full(y.size,x),y.ravel(),z.ravel()-.3]))
    raw=np.concatenate(walls).astype(np.float32);pose=np.zeros((len(t),13));pose[:,6]=1
    return dict(frame_stamps_ns=(1e9+t*1e9).astype(np.int64),frame_interpolated_odom=pose,
                frame_offsets=np.arange(len(t)+1)*len(raw),local_xyz=np.tile(raw,(len(t),1)),
                odom_stamps_ns=(1e9+t*1e9).astype(np.int64),odom_values=pose.copy())


@pytest.mark.parametrize('x',[5.2,5.95])
def test_known_ten_cm_normal_change_in_memory_copy_is_detected_across_cell_boundary(x):
    original=fixture(x);before=original['local_xyz'].copy();changed={k:v.copy() for k,v in original.items()}
    # Deliberately modify only the synthetic in-memory copy's latter half.
    changed['local_xyz'][changed['frame_offsets'][len(changed['frame_stamps_ns'])//2]:,0]+=.10
    result=analyse(changed,dict(DEFAULTS))
    assert result['status']=='NORMAL_RESIDUAL_CHANGE_DETECTED'
    assert max(p['maximum_absolute_normal_change_m'] for p in result['patches'])>.09
    assert np.array_equal(original['local_xyz'],before)
    assert result['no_alignment_applied'] and not result['reference_model_used']


def test_single_visit_cannot_be_a_temporal_pass():
    result=analyse(fixture(repeat=False),dict(DEFAULTS))
    assert result['status']=='INCONCLUSIVE'
    assert result['comparable_fixed_patches']==0


def test_no_shift_baseline_is_scoped_to_observed_wall_normals():
    result=analyse(fixture(),dict(DEFAULTS))
    assert result['status']=='NO_LARGE_NORMAL_CHANGE_IN_OBSERVED_PATCHES'
    assert len(result['patches'])==3
    assert all(p['maximum_absolute_normal_change_m']<1e-6 for p in result['patches'])


def test_large_disappearance_is_inconclusive_not_no_change():
    original=fixture();changed={k:v.copy() for k,v in original.items()}
    changed['local_xyz'][changed['frame_offsets'][4]:,0]+=.8
    result=analyse(changed,dict(DEFAULTS))
    assert result['status'].startswith('INCONCLUSIVE')
    assert result['repeated_lost_support_patch_cells']
    assert result['unresolved_patches']


def test_static_interpolation_rejects_long_bracket_and_unsettled_pose():
    params=dict(DEFAULTS);stamps=np.array([2_000_000_000,2_010_000_000],np.int64);values=np.zeros((2,13));values[:,6]=1
    out,reason=interpolate_pose(stamps,values,np.array([0,0]),2_005_000_000,params)
    assert out is not None and not reason
    out,reason=interpolate_pose(stamps,values,np.array([1_500_000_000]*2),2_005_000_000,params)
    assert out is None and reason=='not_settled'
    stamps[1]=2_100_000_000
    out,reason=interpolate_pose(stamps,values,np.array([0,0]),2_050_000_000,params)
    assert out is None and reason=='odom_bracket_gap'


@pytest.mark.parametrize('state,expected', [('SAVING','acquisition_finished'),('VALIDATING','acquisition_finished'),('COMPLETED','acquisition_finished'),('CANCELLED','run_cancelled:test'),('FAILED','run_failed:test')])
def test_recorder_follows_owned_run_lifecycle(tmp_path,state,expected):
    from record_static_frames import Recorder
    node=object.__new__(Recorder)
    runtime=tmp_path/'runtime.json';run=tmp_path/'run.json'
    runtime.write_text(json.dumps({'state':'RUNNING','generation':'g'}));run.write_text(json.dumps({'run_id':'r','state':state,'reason':'test'}))
    node.args=SimpleNamespace(runtime_status=runtime,run_status=run,run_id='r',output=tmp_path)
    node.last_monitor=0.;node.last_flush=time.monotonic();node.runtime_generation='g';node.failure=None;node.close_requested=None
    node.monitor()
    assert (node.failure or node.close_requested)==expected


def live_capture_set(folder):
    from temporal_consistency import write_capture,sha
    original=fixture();source={'kind':'live_subscriber_samples','run_id':'r','runtime_generation':'g',
        'helper_script_sha256':sha(Path(__file__).with_name('temporal_consistency.py')) if (Path(__file__).with_name('temporal_consistency.py')).exists() else sha(Path(__file__).parents[1]/'validation'/'temporal_consistency.py'),
        'recorder_script_sha256':'a'*64,'ros_domain_id':'178','ros_distro':'test'}
    paths=[];parts=[];total_odom=[]
    for index in range(2):
        frames=[];od=[];stamps=[]
        for i in range(index*6,(index+1)*6):
            t=int(original['frame_stamps_ns'][i]);target=t-40_000_000;bracket=[target-5_000_000,target+5_000_000]
            pose=original['frame_interpolated_odom'][i]
            frames.append(dict(stamp=t,received=t,target=target,bracket=bracket,pose=pose,xyz=original['local_xyz'][original['frame_offsets'][i]:original['frame_offsets'][i+1]]))
            stamps+=bracket;od.extend([pose,pose])
        path=folder/f'chunk_{index:04d}.npz';paths.append(path);total_odom.extend(stamps)
        metadata=write_capture(path,stamps,stamps,od,frames,dict(DEFAULTS),{'stationary_frames':(index+1)*6,'odom_received':(index+1)*12},source)
        parts.append({'name':path.name,'sha256':metadata['capture_sha256'],'bytes':path.stat().st_size,'odom_samples':12,'stationary_frames':6})
    manifest={'schema_version':1,'capture_state':'CLOSED','close_reason':'acquisition_finished','run_id':'r','runtime_generation':'g',
              'parameters':dict(DEFAULTS),'sources':source,'chunks':parts,'total_bytes':sum(p['bytes'] for p in parts),'statistics':{'stationary_frames':12,'odom_received':24},
              'first_frame_stamp_ns':int(original['frame_stamps_ns'][0]),'last_frame_stamp_ns':int(original['frame_stamps_ns'][-1]),
              'first_odom_stamp_ns':total_odom[0],'last_odom_stamp_ns':total_odom[-1]}
    (folder/'capture_manifest.json').write_text(json.dumps(manifest))
    return paths,manifest


def test_complete_manifest_allows_order_independent_input_and_scoped_result(tmp_path):
    from temporal_consistency import load_captures
    paths,_=live_capture_set(tmp_path);data,evidence=load_captures(paths[::-1],return_evidence=True)
    assert evidence['status']=='VERIFIED_COMPLETE_CAPTURE_SET'
    assert analyse(data,dict(DEFAULTS))['status']=='NO_LARGE_NORMAL_CHANGE_IN_OBSERVED_PATCHES'


@pytest.mark.parametrize('damage',['omitted_input','duplicate_input','missing_file','hash_corrupt','missing_manifest','incomplete_manifest','early_stop','duplicate_manifest_chunk','missing_middle_sequence','wrong_run','wrong_generation','wrong_size','wrong_count','wrong_span','mixed_run','missing_sidecar','changed_parameters','stray_chunk','invalid_offsets','wrong_manifest_digest','malformed_archive','duplicate_frame_stamp'])
def test_incomplete_or_mixed_capture_cannot_claim_no_change(tmp_path,damage):
    from temporal_consistency import load_captures,CaptureIntegrityError,sha
    paths,manifest=live_capture_set(tmp_path);manifest_path=tmp_path/'capture_manifest.json'
    if damage=='omitted_input':paths=paths[:-1]
    elif damage=='duplicate_input':paths.append(paths[-1])
    elif damage=='missing_file':paths[-1].unlink()
    elif damage=='hash_corrupt':paths[-1].write_bytes(paths[-1].read_bytes()+b'tampered')
    elif damage=='missing_manifest':manifest_path.unlink()
    elif damage=='incomplete_manifest':manifest['capture_state']='INCOMPLETE'
    elif damage=='early_stop':manifest['close_reason']='operator_stopped'
    elif damage=='duplicate_manifest_chunk':manifest['chunks'].append(dict(manifest['chunks'][-1]))
    elif damage=='missing_middle_sequence':manifest['chunks'][1]['name']='chunk_0002.npz'
    elif damage=='wrong_run':manifest['run_id']='other'
    elif damage=='wrong_generation':manifest['runtime_generation']='other'
    elif damage=='wrong_manifest_digest':manifest['chunks'][0]['sha256']='0'*64
    elif damage=='wrong_size':manifest['chunks'][0]['bytes']+=1
    elif damage=='wrong_count':manifest['chunks'][0]['stationary_frames']-=1
    elif damage=='wrong_span':manifest['last_frame_stamp_ns']-=1
    elif damage=='stray_chunk':(tmp_path/'chunk_0002.npz').write_bytes(paths[-1].read_bytes())
    elif damage=='missing_sidecar':paths[-1].with_suffix('.json').unlink()
    elif damage in ('mixed_run','changed_parameters'):
        sidecar=paths[-1].with_suffix('.json');value=json.loads(sidecar.read_text())
        if damage=='mixed_run':value['source']['run_id']='other'
        else:value['parameters']['pose_offset_s']=0.
        sidecar.write_text(json.dumps(value))
    elif damage in ('invalid_offsets','malformed_archive','duplicate_frame_stamp'):
        path=paths[-1]
        with np.load(path) as bundle:arrays={k:bundle[k] for k in bundle.files}
        if damage=='invalid_offsets':arrays['frame_offsets'][-1]-=1
        elif damage=='duplicate_frame_stamp':arrays['frame_stamps_ns'][1]=arrays['frame_stamps_ns'][0]
        np.savez_compressed(path,**arrays)
        if damage=='malformed_archive':path.write_bytes(b'PK'+b'not a valid zip file')
        digest=sha(path);manifest['chunks'][-1]['sha256']=digest;manifest['chunks'][-1]['bytes']=path.stat().st_size
        sidecar=path.with_suffix('.json');value=json.loads(sidecar.read_text());value['capture_sha256']=digest;sidecar.write_text(json.dumps(value))
    if damage not in ('missing_manifest',):manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CaptureIntegrityError):load_captures(paths)


def test_cli_corrupt_capture_writes_inconclusive_and_nonzero(tmp_path,monkeypatch,capsys):
    import temporal_consistency
    paths,_=live_capture_set(tmp_path);paths[-1].unlink();output=tmp_path/'result.json'
    monkeypatch.setattr(sys,'argv',['temporal_consistency.py','analyse',*[str(p) for p in paths],'--output',str(output)])
    assert temporal_consistency.main()==2
    assert json.loads(output.read_text())['status']=='INCONCLUSIVE_INPUT_INTEGRITY'


@pytest.mark.parametrize('budget',[3000,12000,20000,2999,20001,True,1.5])
def test_capture_resolution_is_declared_without_relaxing_review_parameters(tmp_path,budget):
    from temporal_consistency import load_captures,CaptureIntegrityError
    paths,manifest=live_capture_set(tmp_path)
    for path in paths:
        sidecar=path.with_suffix('.json');value=json.loads(sidecar.read_text());value['parameters']['points_per_frame']=budget;sidecar.write_text(json.dumps(value))
    manifest['parameters']['points_per_frame']=budget;(tmp_path/'capture_manifest.json').write_text(json.dumps(manifest))
    if type(budget) is int and budget in (3000,12000,20000):
        data,evidence=load_captures(paths,return_evidence=True)
        assert evidence['parameters']['points_per_frame']==budget
        assert evidence['parameters']['normal_change_review_m']==DEFAULTS['normal_change_review_m']
        assert evidence['parameters']['thickness_change_review_m']==DEFAULTS['thickness_change_review_m']
        assert analyse(data,evidence['parameters'])['status']=='NO_LARGE_NORMAL_CHANGE_IN_OBSERVED_PATCHES'
    else:
        with pytest.raises(CaptureIntegrityError):load_captures(paths)
