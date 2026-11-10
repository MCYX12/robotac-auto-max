#!/usr/bin/env python3
"""Read-only repeated-wall normal consistency, using measured points only.
No scene reference, ICP, alignment, map rewriting or global zero-drift claim.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DEFAULTS = dict(pose_offset_s=-.04, linear_limit=.04, angular_limit=.08,
                settle_s=1., maximum_odom_bracket_s=.02, sample_hz=2.,
                points_per_frame=3000, minimum_range_m=.6, maximum_range_m=20.,
                lidar_translation_m=[0.,0.,.3], cell_size_m=1.,
                normal_capture_band_m=.20, viewpoint_distance_m=1.5,
                viewpoint_yaw_difference_deg=25., minimum_revisit_separation_s=30.,
                minimum_total_span_s=60., normal_change_review_m=.01,
                thickness_change_review_m=.02, minimum_comparable_patches=3)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def stamp_ns(message):
    return int(message.header.stamp.sec)*1_000_000_000+int(message.header.stamp.nanosec)


def odom_values(message):
    p=message.pose.pose;v=message.twist.twist
    return [p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w,
            v.linear.x,v.linear.y,v.linear.z,v.angular.x,v.angular.y,v.angular.z]


def local_points(message, maximum=3000, minimum_range=.6, maximum_range=20.):
    fields={f.name:f for f in message.fields};formats=[];offsets=[]
    if message.row_step<message.width*message.point_step or len(message.data)!=message.row_step*message.height:
        raise ValueError('invalid point payload layout')
    for axis in 'xyz':
        field=fields[axis]
        if field.count!=1 or field.datatype not in (7,8):raise ValueError('invalid XYZ field')
        size=4 if field.datatype==7 else 8
        if field.offset+size>message.point_step:raise ValueError('invalid XYZ offset')
        formats.append(('>' if message.is_bigendian else '<')+'f'+str(size));offsets.append(field.offset)
    dtype=np.dtype(dict(names=list('xyz'),formats=formats,offsets=offsets,itemsize=message.point_step))
    a=np.ndarray((message.height,message.width),dtype=dtype,buffer=message.data,strides=(message.row_step,message.point_step))
    xyz=np.column_stack([a[k].ravel() for k in 'xyz']).astype(np.float32)
    xyz=xyz[np.isfinite(xyz).all(axis=1)]
    ranges=np.linalg.norm(xyz,axis=1);xyz=xyz[(ranges>minimum_range)&(ranges<maximum_range)]
    if len(xyz)>1:xyz=xyz[np.r_[True,np.any(np.diff(xyz,axis=0)!=0,axis=1)]]
    if len(xyz)>maximum:xyz=xyz[np.linspace(0,len(xyz)-1,maximum,dtype=int)]
    return xyz


def stationary_since(stamps,values,params):
    stable=(np.linalg.norm(values[:,7:10],axis=1)<=params['linear_limit'])&(np.linalg.norm(values[:,10:13],axis=1)<=params['angular_limit'])
    since=np.full(len(stamps),-1,dtype=np.int64);begin=-1
    for i in range(len(stamps)):
        if not stable[i]:begin=-1
        elif begin<0 or (i and stamps[i]-stamps[i-1]>100_000_000):begin=int(stamps[i])
        since[i]=begin
    return since


def interpolate_pose(stamps,values,since,target,params):
    after=int(np.searchsorted(stamps,target,side='left'))
    if after==0 or after>=len(stamps):return None,'outside_odom'
    before=after-1;a=int(stamps[before]);b=int(stamps[after])
    if max(target-a,b-target)>round(params['maximum_odom_bracket_s']*1e9):return None,'odom_bracket_gap'
    if since[before]<0 or since[after]<0 or target-int(since[before])<round(params['settle_s']*1e9):return None,'not_settled'
    fraction=(target-a)/(b-a)
    pose=values[before]*(1-fraction)+values[after]*fraction
    pose[3:7]=Slerp([0.,1.],Rotation.from_quat(values[[before,after],3:7]))([fraction]).as_quat()[0]
    return (pose,[a,b]),''


def write_capture(path,odom_stamps,odom_receive,odom,frames,params,stats,source):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    offsets=np.r_[0,np.cumsum([len(f['xyz']) for f in frames])].astype(np.int64)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,odom_stamps_ns=np.asarray(odom_stamps,np.int64),odom_receive_ns=np.asarray(odom_receive,np.int64),odom_values=np.asarray(odom,np.float64).reshape(-1,13),
            frame_stamps_ns=np.asarray([f['stamp'] for f in frames],np.int64),frame_receive_ns=np.asarray([f['received'] for f in frames],np.int64),
            frame_pose_time_ns=np.asarray([f['target'] for f in frames],np.int64),frame_odom_bracket_ns=np.asarray([f['bracket'] for f in frames],np.int64).reshape(-1,2),
            frame_interpolated_odom=np.asarray([f['pose'] for f in frames],np.float64).reshape(-1,13),frame_offsets=offsets,
            local_xyz=np.concatenate([f['xyz'] for f in frames]) if frames else np.empty((0,3),np.float32))
    os.replace(temporary,path)
    metadata={'schema_version':1,'source':source,'parameters':params,'statistics':stats,'capture_sha256':sha(path),'script_sha256':sha(__file__),
              'meaning':'Raw local XYZ samples and actual sensor stamps; interpolated odometry recorded explicitly. No registration or scene-reference points.'}
    path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    return metadata


def extract_bag(bag,output,params):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    bag=Path(bag).resolve();before=bag.stat();db=sqlite3.connect('file:'+str(bag)+'?mode=ro',uri=True)
    topics={name:(i,get_message(typ)) for i,name,typ in db.execute('select id,name,type from topics')}
    oid,otype=topics['/odom/mujoco_odom'];lid,ltype=topics['/front_lidar']
    stamps=[];receive=[];values=[];stats=defaultdict(int)
    for received,payload in db.execute('select timestamp,data from messages where topic_id=? order by timestamp',(oid,)):
        msg=deserialize_message(payload,otype);t=stamp_ns(msg);v=np.asarray(odom_values(msg))
        stats['odom_received']+=1
        if t<=0 or not np.isfinite(v).all() or not .5<float(v[3:7]@v[3:7])<1.5:stats['invalid_odom']+=1;continue
        if stamps and t<=stamps[-1]:stats['nonincreasing_odom_stamp']+=1;continue
        stamps.append(t);receive.append(received);values.append(v)
    stamps=np.asarray(stamps,np.int64);values=np.asarray(values)
    if len(stamps)<2:raise ValueError('insufficient valid odometry')
    since=stationary_since(stamps,values,params)
    print('odometry_loaded',len(stamps),flush=True)
    candidates=[];last=-10**30;interval=round(1e9/params['sample_hz'])
    for rowid,received in db.execute('select id,timestamp from messages where topic_id=? order by timestamp',(lid,)):
        stats['lidar_received']+=1
        if received-last>=interval:candidates.append((rowid,received));last=received
    frames=[]
    for index,(rowid,received) in enumerate(candidates):
        payload=db.execute('select data from messages where id=?',(rowid,)).fetchone()[0]
        msg=deserialize_message(payload,ltype);t=stamp_ns(msg);target=t+round(params['pose_offset_s']*1e9);stats['lidar_rate_selected']+=1
        matched,reason=interpolate_pose(stamps,values,since,target,params)
        if matched is None:stats[reason]+=1;continue
        pose,bracket=matched
        try:xyz=local_points(msg,params['points_per_frame'],params['minimum_range_m'],params['maximum_range_m'])
        except (ValueError,KeyError,TypeError):stats['invalid_cloud']+=1;continue
        if len(xyz)<100:stats['insufficient_local_points']+=1;continue
        frames.append(dict(stamp=t,received=received,target=target,pose=pose,bracket=bracket,xyz=xyz));stats['stationary_frames']+=1
        if index and index%500==0:print('frame_progress',index,'/',len(candidates),'accepted',len(frames),flush=True)
    db.close()
    after=bag.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('source bag changed during extraction')
    source={'kind':'readonly_rosbag2_sqlite','path':str(bag),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'sha256':sha(bag)}
    result=write_capture(output,stamps,receive,values,frames,params,dict(stats),source)
    print('capture_written',json.dumps(result['statistics']),flush=True)
    return result


class CaptureIntegrityError(ValueError):
    """Evidence is incomplete or inconsistent; no temporal conclusion is permitted."""


def _require(condition, reason):
    if not condition:raise CaptureIntegrityError(reason)


def _read_evidence_json(path, snapshots):
    try:
        raw=path.read_bytes();value=json.loads(raw)
    except (OSError,ValueError) as error:raise CaptureIntegrityError(f'{path.name}: missing/invalid metadata: {error}') from error
    _require(isinstance(value,dict) and value.get('schema_version')==1,f'{path.name}: invalid metadata schema')
    snapshots[path]=raw
    return value


def _validate_arrays(data, label, params):
    shapes={'odom_stamps_ns':(-1,), 'odom_receive_ns':(-1,), 'odom_values':(-1,13),
            'frame_stamps_ns':(-1,), 'frame_receive_ns':(-1,), 'frame_pose_time_ns':(-1,),
            'frame_odom_bracket_ns':(-1,2), 'frame_interpolated_odom':(-1,13),
            'frame_offsets':(-1,), 'local_xyz':(-1,3)}
    _require(set(data)==set(shapes),label+': unexpected/missing arrays')
    for key,shape in shapes.items():
        array=data[key]
        _require(array.ndim==len(shape) and array.shape[1:]==shape[1:],label+': invalid shape '+key)
        expected='iu' if key.endswith('_ns') or key=='frame_offsets' else 'f'
        _require(array.dtype.kind in expected and np.isfinite(array).all(),label+': invalid numeric values '+key)
    n=len(data['odom_stamps_ns']);m=len(data['frame_stamps_ns']);offsets=data['frame_offsets']
    _require(len(data['odom_receive_ns'])==len(data['odom_values'])==n,label+': odometry length mismatch')
    _require(all(len(data[k])==m for k in ('frame_receive_ns','frame_pose_time_ns','frame_odom_bracket_ns','frame_interpolated_odom')),label+': frame length mismatch')
    _require(len(offsets)==m+1 and offsets[0]==0 and offsets[-1]==len(data['local_xyz']) and np.all(np.diff(offsets)>=100) and np.all(np.diff(offsets)<=params['points_per_frame']),label+': invalid frame point offsets')
    for key in ('odom_stamps_ns','frame_stamps_ns'):
        values=data[key];_require(np.all(values>0) and np.all(np.diff(values)>0),label+': duplicate/reordered sensor timestamps '+key)
    for key in ('odom_values','frame_interpolated_odom'):
        norm=np.sum(data[key][:,3:7]**2,axis=1)
        _require(np.all((norm>.5)&(norm<1.5)),label+': invalid quaternion '+key)
    target=data['frame_pose_time_ns'];brackets=data['frame_odom_bracket_ns']
    _require(np.array_equal(target,data['frame_stamps_ns']+round(params['pose_offset_s']*1e9)),label+': pose offset differs from recorded parameters')
    _require(np.all((brackets[:,0]<brackets[:,1])&(brackets[:,0]<=target)&(target<=brackets[:,1])) and np.all(np.abs(brackets-target[:,None])<=round(params['maximum_odom_bracket_s']*1e9)),label+': invalid interpolation brackets')


def load_captures(paths, return_evidence=False):
    """Accept a complete single recording, never an arbitrary subset of live chunks.

    Digests detect accidental corruption, not an adversary rewriting all evidence.
    Standalone bag extracts require their original sidecar; live captures additionally
    require a closed manifest, one run/generation, and its exact contiguous chunk set.
    """
    paths=[Path(path).resolve() for path in paths]
    _require(bool(paths),'no capture inputs')
    _require(len(set(paths))==len(paths),'duplicate capture input')
    snapshots={};metadata=[_read_evidence_json(p.with_suffix('.json'),snapshots) for p in paths]
    source=metadata[0].get('source');params=metadata[0].get('parameters')
    _require(isinstance(source,dict),'missing source identity')
    _require(isinstance(params,dict),'missing capture parameters')
    # Sampling density is evidence resolution, not an acceptance tolerance.
    # Retain exact calibration, stability, revisit and review thresholds.
    budget=params.get('points_per_frame')
    _require(type(budget) is int and budget in (3000,12000,20000),'unsupported capture sampling budget')
    expected=dict(DEFAULTS);expected['points_per_frame']=budget
    _require(params==expected,'recorded parameters differ from analysis parameters')
    _require(all(m.get('source')==source and m.get('parameters')==params for m in metadata),'mixed source/run/generation/parameters')
    kind=source.get('kind');manifest=None;entries={}
    if kind=='live_subscriber_samples':
        _require(len({p.parent for p in paths})==1,'live chunks must belong to one recording directory')
        parent=paths[0].parent;manifest=_read_evidence_json(parent/'capture_manifest.json',snapshots)
        _require(manifest.get('capture_state')=='CLOSED','capture did not close cleanly')
        _require(manifest.get('close_reason') in ('acquisition_finished','runtime_stopped'),'recording ended before a confirmed acquisition lifecycle boundary')
        _require(source.get('run_id') not in (None,'','unspecified') and source.get('runtime_generation') not in (None,''),'missing live run/generation identity')
        _require(manifest.get('sources')==source and manifest.get('parameters')==params and manifest.get('run_id')==source['run_id'] and manifest.get('runtime_generation')==source['runtime_generation'],'manifest source/run/generation/parameters mismatch')
        chunks=manifest.get('chunks');_require(isinstance(chunks,list) and bool(chunks),'manifest has no chunks')
        for index,entry in enumerate(chunks):
            _require(isinstance(entry,dict) and entry.get('name')==f'chunk_{index:04d}.npz','duplicate, missing or reordered manifest chunk sequence')
            entries[entry['name']]=entry
        _require({p.name for p in paths}==set(entries),'selected inputs omit or add manifest chunks')
        _require({p.name for p in parent.glob('chunk_*.npz')}==set(entries),'on-disk chunk set differs from manifest')
        pairs=sorted(zip(paths,metadata),key=lambda item:item[0].name);paths=[x[0] for x in pairs];metadata=[x[1] for x in pairs]
        _require(all(m.get('script_sha256')==source.get('helper_script_sha256') for m in metadata),'helper source changed during recording')
    else:
        _require(kind=='readonly_rosbag2_sqlite' and len(paths)==1,'only one standalone bag extraction may be analysed at once')
        _require(isinstance(source.get('sha256'),str) and len(source['sha256'])==64,'missing original bag digest')
    bundles=[];inputs=[]
    for path,meta in zip(paths,metadata):
        try:
            digest=sha(path);size=path.stat().st_size
            _require(digest==meta.get('capture_sha256'),path.name+': sidecar capture hash mismatch')
            if manifest:
                entry=entries[path.name]
                _require(entry.get('sha256')==digest and entry.get('bytes')==size,path.name+': manifest hash/size mismatch')
            with np.load(path,allow_pickle=False) as data:bundle={k:data[k] for k in data.files}
            _require(sha(path)==digest,path.name+': capture changed while reading')
        except (OSError,KeyError,ValueError,EOFError,zipfile.BadZipFile) as error:
            if isinstance(error,CaptureIntegrityError):raise
            raise CaptureIntegrityError(f'{path.name}: unreadable capture: {error}') from error
        _validate_arrays(bundle,path.name,params)
        if manifest:
            _require(entry.get('odom_samples')==len(bundle['odom_stamps_ns']) and entry.get('stationary_frames')==len(bundle['frame_stamps_ns']),path.name+': manifest sample count mismatch')
        bundles.append(bundle);inputs.append({'path':str(path),'sha256':digest,'bytes':size})
    arrays={key:np.concatenate([d[key] for d in bundles]) for key in bundles[0] if key!='frame_offsets'}
    counts=np.concatenate([np.diff(d['frame_offsets']) for d in bundles]);arrays['frame_offsets']=np.r_[0,np.cumsum(counts)]
    _validate_arrays(arrays,'combined capture',params)
    _require(len(arrays['odom_stamps_ns'])>=2,'insufficient valid odometry evidence')
    if manifest:
        _require(manifest.get('total_bytes')==sum(i['bytes'] for i in inputs),'manifest total size mismatch')
        stats=manifest.get('statistics',{})
        _require(isinstance(stats,dict),'invalid manifest statistics')
        _require(stats.get('stationary_frames',0)==len(arrays['frame_stamps_ns']),'manifest total stationary frame count mismatch')
        for label,key in (('frame','frame_stamps_ns'),('odom','odom_stamps_ns')):
            values=arrays[key]
            _require(manifest.get(f'first_{label}_stamp_ns')==(int(values[0]) if len(values) else None) and manifest.get(f'last_{label}_stamp_ns')==(int(values[-1]) if len(values) else None),'manifest '+label+' time coverage mismatch')
    for path,raw in snapshots.items():
        _require(path.read_bytes()==raw,path.name+': metadata changed during analysis load')
    evidence={'status':'VERIFIED_COMPLETE_CAPTURE_SET','kind':kind,'source':source,'inputs':inputs,'parameters':params,
              'manifest_sha256':hashlib.sha256(snapshots[paths[0].parent/'capture_manifest.json']).hexdigest() if manifest else None,
              'scope':'Recorded files and declared chunk set verified; this does not prove complete scene coverage or zero drift.'}
    return (arrays,evidence) if return_evidence else arrays


def fit_line(points,seed):
    if len(points)<80:return None
    rng=np.random.default_rng(seed);xy=points[:,:2];best=None
    for _ in range(100):
        a,b=xy[rng.choice(len(xy),2,replace=False)];v=b-a;length=np.linalg.norm(v)
        if length<.2:continue
        normal=np.array([-v[1],v[0]])/length;d=-normal@a
        mask=np.abs(xy@normal+d)<.015
        if best is None or mask.sum()>best.sum():best=mask
    if best is None or best.sum()<80 or best.mean()<.65:return None
    inliers=points[best];center=np.mean(inliers[:,:2],axis=0)
    _,_,vh=np.linalg.svd(inliers[:,:2]-center,full_matrices=False);normal=vh[-1]
    if normal[0]<0 or (abs(normal[0])<1e-12 and normal[1]<0):normal=-normal
    tangent=np.array([-normal[1],normal[0]]);d=-normal@center
    u=inliers[:,:2]@tangent;u_bounds=np.percentile(u,[2,98]);z_bounds=np.percentile(inliers[:,2],[5,95])
    if u_bounds[1]-u_bounds[0]<.35 or z_bounds[1]-z_bounds[0]<.4:return None
    residual=inliers[:,:2]@normal+d
    if np.percentile(np.abs(residual-np.median(residual)),95)>.01:return None
    return normal,float(d),tangent,u_bounds,z_bounds


def analyse(data,params):
    stamps=data['frame_stamps_ns'];poses=data['frame_interpolated_odom'];offsets=data['frame_offsets'];local=data['local_xyz']
    report={'status':'INCONCLUSIVE','observable':'Repeated measured wall-normal signed residual and thickness in fixed native odometry coordinates only',
            'no_alignment_applied':True,'reference_model_used':False,'parameters':params,'accepted_stationary_frames':len(stamps),
            'limitations':['Cannot prove zero global drift or observe every tangential/rotation degree of freedom.','Changes can arise from pose error, timing/extrinsics or view-dependent returns; not automatically attributed to SLAM drift.','Large displacement can leave a frozen patch; lost support is reported, never treated as agreement.','Results concern sampled stationary LiDAR structure, not moving floor/depth mapping or unseen surfaces.'],
            'patches':[],'unresolved_patches':[],'eligibility':{},'odometry':{}}
    od=data['odom_values'];ot=data['odom_stamps_ns'];order=np.argsort(ot);ot=ot[order];od=od[order];keep=np.r_[True,np.diff(ot)>0];ot=ot[keep];od=od[keep]
    if len(ot)>1:
        dt=np.diff(ot)*1e-9;distance=np.linalg.norm(np.diff(od[:,:3],axis=0),axis=1)
        angle=(Rotation.from_quat(od[:-1,3:7]).inv()*Rotation.from_quat(od[1:,3:7])).magnitude()
        jump=(distance>.35+3*dt)|(angle>.35+3*dt)
        report['odometry']={'samples':len(ot),'span_s':float((ot[-1]-ot[0])*1e-9),'time_gap_over_100ms_count':int(np.sum(dt>.1)),'position_or_rotation_jump_count':int(jump.sum()),'maximum_step_m':float(distance.max()),'maximum_step_angle_rad':float(angle.max()),'jump_times_ns':ot[1:][jump].tolist()[:100]}
    if len(stamps)<6:return report
    rotations=Rotation.from_quat(poses[:,3:7]);yaw=rotations.as_euler('xyz')[:,2]
    groups=defaultdict(list);world_frames=[];start=int(stamps[0]);times=(stamps-start)*1e-9
    for frame in range(len(stamps)):
        xyz=rotations[frame].apply(local[offsets[frame]:offsets[frame+1]]+params['lidar_translation_m'])+poses[frame,:3]
        xyz=xyz[(xyz[:,2]>.4)&(xyz[:,2]<2.7)]
        world_frames.append(xyz)
        cells=np.floor(xyz[:,:2]/params['cell_size_m']).astype(int)
        unique,inverse=np.unique(cells,axis=0,return_inverse=True)
        for i,cell in enumerate(unique):
            points=xyz[inverse==i]
            if len(points)>=8:groups[tuple(cell)].append((frame,points))
    stats=defaultdict(int);stats['measured_spatial_cells']=len(groups)
    for cell,observations in sorted(groups.items()):
        if len(observations)<3:stats['too_few_baseline_frame_observations']+=1;continue
        observations.sort(key=lambda item:stamps[item[0]])
        first_time=times[observations[0][0]];baseline=[]
        for frame,points in observations:
            if times[frame]-first_time>5.:break
            baseline.append((frame,points))
        if len(baseline)<3:stats['insufficient_baseline_frames']+=1;continue
        model=fit_line(np.concatenate([v for _,v in baseline]),abs(cell[0]*73856093+cell[1]*19349663)%(2**32))
        if model is None:stats['not_a_stable_single_vertical_plane']+=1;continue
        normal,d,tangent,ub,zb=model;baseline_ids=np.array([f for f,_ in baseline]);measurements=[];unmatched=0;lost=0
        for frame,points in enumerate(world_frames):
            if times[frame]<first_time:continue
            displacement=np.linalg.norm(poses[baseline_ids,:2]-poses[frame,:2],axis=1)
            yaw_difference=np.abs(np.arctan2(np.sin(yaw[baseline_ids]-yaw[frame]),np.cos(yaw[baseline_ids]-yaw[frame])))
            if not np.any((displacement<=params['viewpoint_distance_m'])&(yaw_difference<=math.radians(params['viewpoint_yaw_difference_deg']))):
                unmatched+=1;continue
            u=points[:,:2]@tangent;residual=points[:,:2]@normal+d
            mask=(u>=ub[0])&(u<=ub[1])&(points[:,2]>=zb[0])&(points[:,2]<=zb[1])&(np.abs(residual)<=.5)
            support=mask.sum();within=mask&(np.abs(residual)<=params['normal_capture_band_m'])
            if within.sum()<15 or (support and within.sum()/support<.65):lost+=1;continue
            r=residual[within];median=float(np.median(r));q=np.percentile(r,[5,95])
            measurements.append({'frame':frame,'time_s':float(times[frame]),'stamp_ns':int(stamps[frame]),'points':int(len(r)),'signed_median_m':median,'centered_abs_p95_m':float(np.percentile(np.abs(r-median),95)),'thickness_p95_p05_m':float(q[1]-q[0])})
        unresolved={'cell':[int(c) for c in cell],'normal_xy':normal.tolist(),'baseline_start_s':float(first_time),'baseline_frames':len(baseline),'viewpoint_unmatched_frames':unmatched,'lost_normal_support_frames':lost,'matched_supported_frames':len(measurements)}
        if len(measurements)<6:
            stats['insufficient_matched_measurements']+=1;unresolved['reason']='insufficient_matched_measurements';report['unresolved_patches'].append(unresolved);continue
        visits=[]
        for item in measurements:
            if not visits or item['time_s']-visits[-1][-1]['time_s']>8.:visits.append([])
            visits[-1].append(item)
        visits=[v for v in visits if len(v)>=3]
        summaries=[dict(start_s=v[0]['time_s'],end_s=v[-1]['time_s'],frames=len(v),points=sum(x['points'] for x in v),signed_median_m=float(np.median([x['signed_median_m'] for x in v])),thickness_m=float(np.median([x['thickness_p95_p05_m'] for x in v]))) for v in visits]
        if len(summaries)<2 or summaries[-1]['start_s']-summaries[0]['start_s']<params['minimum_total_span_s']:
            stats['insufficient_revisit_time_span']+=1;unresolved['reason']='insufficient_revisit_time_span';report['unresolved_patches'].append(unresolved);continue
        early=summaries[0];late=[v for v in summaries[1:] if v['start_s']-early['end_s']>=params['minimum_revisit_separation_s']]
        if not late:
            stats['no_separated_revisit']+=1;unresolved['reason']='no_separated_revisit';report['unresolved_patches'].append(unresolved);continue
        changes=[v['signed_median_m']-early['signed_median_m'] for v in late]
        thickness=[v['thickness_m']-early['thickness_m'] for v in late]
        patch={'cell':[int(c) for c in cell],'normal_xy':normal.tolist(),'plane_d_m':d,'tangent_bounds_m':ub.tolist(),'height_bounds_m':zb.tolist(),'baseline_frames':[int(f) for f in baseline_ids],
               'observation_span_s':summaries[-1]['end_s']-early['start_s'],'matched_visits':summaries,'measurements':measurements,'viewpoint_unmatched_frames':unmatched,'lost_normal_support_frames':lost,
               'early_late_signed_change_m':changes[-1],'maximum_absolute_normal_change_m':float(max(abs(x) for x in changes)),
               'early_late_thickness_change_m':thickness[-1],'maximum_thickness_increase_m':float(max(thickness)),
               'normal_change_review':any(abs(x)>params['normal_change_review_m'] for x in changes),'thickness_change_review':any(x>params['thickness_change_review_m'] for x in thickness)}
        report['patches'].append(patch)
    report['eligibility']=dict(stats);patches=report['patches'];report['comparable_fixed_patches']=len(patches)
    report['fixed_patch_support_area_sum_m2']=float(sum((p['tangent_bounds_m'][1]-p['tangent_bounds_m'][0])*(p['height_bounds_m'][1]-p['height_bounds_m'][0]) for p in patches))
    if len(patches)>=params['minimum_comparable_patches']:
        flagged=any(p['normal_change_review'] or p['thickness_change_review'] for p in patches) or report['odometry'].get('position_or_rotation_jump_count',0)>0
        report['status']='NORMAL_RESIDUAL_CHANGE_DETECTED' if flagged else 'NO_LARGE_NORMAL_CHANGE_IN_OBSERVED_PATCHES'
        report['normal_change_absolute_percentiles_m']={str(q):float(np.percentile([abs(p['early_late_signed_change_m']) for p in patches],q)) for q in (50,90,95,99)}
    lost_patches=[p['cell'] for p in patches+report['unresolved_patches'] if p.get('lost_normal_support_frames',0)>=3]
    report['repeated_lost_support_patch_cells']=lost_patches
    if lost_patches and report['status']=='NO_LARGE_NORMAL_CHANGE_IN_OBSERVED_PATCHES':report['status']='INCONCLUSIVE_LOST_PATCH_SUPPORT'
    report['inconclusive_reason']='repeated loss of support in matched viewpoints' if report['status']=='INCONCLUSIVE_LOST_PATCH_SUPPORT' else None if len(patches)>=params['minimum_comparable_patches'] else 'fewer than required fixed patches have comparable early/late repeated observations'
    report['support_area_note']='Sum of frozen rectangular patch supports; may overlap and is not a percentage of scene or map coverage.'
    return report


def plot_report(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    patches=sorted(report['patches'],key=lambda p:p['maximum_absolute_normal_change_m'],reverse=True)[:8]
    if not patches:return
    fig,axes=plt.subplots(len(patches),1,figsize=(12,2.1*len(patches)),squeeze=False)
    for ax,p in zip(axes[:,0],patches):
        m=p['measurements'];base=p['matched_visits'][0]['signed_median_m']
        ax.plot([v['time_s']/60 for v in m],[(v['signed_median_m']-base)*1000 for v in m],'.',markersize=3)
        ax.axhline(0,color='gray',lw=.7);ax.axhline(10,color='orange',ls='--',lw=.7);ax.axhline(-10,color='orange',ls='--',lw=.7)
        ax.set_ylabel('normal change (mm)');ax.set_title(f"fixed patch {p['cell']} | normal {np.round(p['normal_xy'],3)} | matched visits {len(p['matched_visits'])}",fontsize=10)
    axes[-1,0].set_xlabel('elapsed minutes within the same recording')
    fig.suptitle(report['status']+'\nMeasured fixed planes; no registration; wall-normal observability only',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(path,dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    extract=sub.add_parser('extract');extract.add_argument('bag',type=Path);extract.add_argument('--output',type=Path,required=True)
    analyse_parser=sub.add_parser('analyse');analyse_parser.add_argument('captures',type=Path,nargs='+');analyse_parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();params=dict(DEFAULTS)
    if args.action=='extract':extract_bag(args.bag,args.output,params);return
    try:
        data,integrity=load_captures(args.captures,return_evidence=True);report=analyse(data,integrity['parameters'])
        report['capture_integrity']=integrity;report['inputs']=integrity['inputs']
    except (CaptureIntegrityError,OSError,ValueError) as error:
        report={'status':'INCONCLUSIVE_INPUT_INTEGRITY','inconclusive_reason':str(error),'patches':[],
                'capture_integrity':{'status':'REJECTED'},'reference_model_used':False,'no_alignment_applied':True}
    report['analysis_script_sha256']=sha(__file__)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');plot_report(report,args.output.with_suffix('.png'))
    print(json.dumps({k:v for k,v in report.items() if k not in ('patches','parameters','inputs')},ensure_ascii=False,indent=2))
    return 2 if report['status']=='INCONCLUSIVE_INPUT_INTEGRITY' else 0


if __name__=='__main__':raise SystemExit(main())
