#!/usr/bin/env python3
"""Passive /robotac/nav_map snapshots bound to one run and runtime generation.
Only subscribes to navigation data; no control/business publishers, services or SDK.
ROS may create its standard parameter_events publisher; rosout/parameter services are disabled.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from nav_msgs.msg import OccupancyGrid

TOPIC='/robotac/nav_map'
MAX_CELLS=16_000_000


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,path)


def read_state(path):
    try:
        result=json.loads(Path(path).read_text())
        return result if isinstance(result,dict) else {}
    except (OSError,ValueError):return {}


def stamp_ns(stamp):
    return int(stamp.sec)*1_000_000_000+int(stamp.nanosec)


def parse_grid(message):
    info=message.info;width=int(info.width);height=int(info.height);resolution=float(info.resolution)
    if not 0<width*height<=MAX_CELLS or width<=0 or height<=0 or len(message.data)!=width*height:
        raise ValueError('invalid_grid_dimensions_or_payload')
    if not math.isfinite(resolution) or resolution<=0:raise ValueError('invalid_resolution')
    position=np.array([info.origin.position.x,info.origin.position.y,info.origin.position.z],np.float64)
    quaternion=np.array([info.origin.orientation.x,info.origin.orientation.y,info.origin.orientation.z,info.origin.orientation.w],np.float64)
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or not .5<float(quaternion@quaternion)<1.5:
        raise ValueError('invalid_origin')
    stamp=stamp_ns(message.header.stamp)
    if stamp<=0 or not message.header.frame_id:raise ValueError('invalid_header')
    grid=np.asarray(message.data)
    if grid.dtype.kind not in 'iu' or np.any(grid<-1) or np.any(grid>100):raise ValueError('invalid_occupancy_values')
    data=grid.astype(np.int8,copy=True).reshape(height,width)
    metadata={'topic':TOPIC,'frame_id':message.header.frame_id,'header_stamp_ns':stamp,
              'map_load_time_ns':stamp_ns(info.map_load_time),'width':width,'height':height,'resolution_m':resolution,
              'origin_position_m':position.tolist(),'origin_quaternion_xyzw':quaternion.tolist(),
              'unknown_cells':int(np.sum(data==-1)),'free_cells':int(np.sum(data==0)),
              'positive_occupancy_cells':int(np.sum(data>0)),'occupied_at_50_cells':int(np.sum(data>=50)),
              'layout':'row-major [height,width], original OccupancyGrid values -1..100; no reclassification or resampling'}
    return data,metadata


def lifecycle(runtime,run,run_id,generation):
    """Return (failure, normal close reason); missing files are handled by caller."""
    if runtime.get('generation')!=generation:return 'runtime_generation_changed',None
    if run.get('run_id')!=run_id:return 'run_id_changed',None
    if runtime.get('state')=='FAILED':return 'runtime_failed:'+str(runtime.get('reason')),None
    if run.get('state') in ('FAILED','CANCELLED'):return 'run_'+run['state'].lower()+':'+str(run.get('reason')),None
    if run.get('state') in ('SAVING','VALIDATING','COMPLETED','NEEDS_REVIEW'):return None,'acquisition_finished'
    if runtime.get('state')=='STOPPED':return None,'runtime_stopped'
    return None,None


class GridRecorder(Node):
    def __init__(self,args):
        runtime=read_state(args.runtime_status);run=read_state(args.run_status)
        if run.get('run_id')!=args.run_id or run.get('state')!='RUNNING' or not runtime.get('generation') or runtime.get('state')!='RUNNING':
            raise RuntimeError('observer must start during the matching RUNNING run with a live runtime generation')
        super().__init__('robotac_navigation_grid_observer',enable_rosout=False,start_parameter_services=False)
        self.args=args;self.generation=runtime['generation'];self.started=time.monotonic();self.started_wall_ns=time.time_ns()
        self.stats=defaultdict(int);self.latest=None;self.last_saved_monotonic=self.started-args.interval;self.parts=[];self.failure=None;self.close_requested=None;self.missing_state_since=None
        self.source={'kind':'live_navigation_grid_subscriber','script_sha256':sha(__file__),'run_id':args.run_id,'runtime_generation':self.generation,
                     'ros_domain_id':os.environ.get('ROS_DOMAIN_ID'),'ros_distro':os.environ.get('ROS_DISTRO'),'topic':TOPIC,
                     'ros_framework_publishers_allowed':['/parameter_events'],'control_or_business_publishers':[],
                     'qos':{'history':'KEEP_LAST','depth':1,'reliability':'RELIABLE','durability':'TRANSIENT_LOCAL'}}
        self.owner={'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'run_id':args.run_id,'generation':self.generation,'script':str(Path(__file__).resolve()),'state':'RECORDING'}
        atomic_json(args.output/'owner.json',self.owner)
        qos=QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid,TOPIC,self.on_grid,qos)

    def check_lifecycle(self):
        runtime=read_state(self.args.runtime_status);run=read_state(self.args.run_status)
        if not runtime or not run:
            if self.missing_state_since is None:self.missing_state_since=time.monotonic()
            if time.monotonic()-self.missing_state_since>5.:self.failure='status_files_unreadable'
            return False
        self.missing_state_since=None;failure,close=lifecycle(runtime,run,self.args.run_id,self.generation)
        self.failure=self.failure or failure;self.close_requested=self.close_requested or close
        return not self.failure and not self.close_requested

    def on_grid(self,message):
        self.stats['received']+=1
        if not self.check_lifecycle():self.stats['rejected_lifecycle']+=1;return
        try:data,metadata=parse_grid(message)
        except (ValueError,TypeError,AttributeError) as error:
            self.stats['invalid_grids']+=1;self.stats['last_invalid_reason']=str(error);return
        if self.latest and metadata['header_stamp_ns']<self.latest[1]['header_stamp_ns']:
            self.failure='navigation_grid_timestamp_regressed';return
        self.stats['accepted']+=1;metadata['reception_sequence']=self.stats['accepted'];self.latest=(data,metadata,time.monotonic(),time.time_ns())

    def save_latest(self,reason):
        if self.latest is None:return
        if shutil.disk_usage(self.args.output).free<512*1024**2:raise RuntimeError('insufficient_disk_space_for_grid_snapshot')
        data,metadata,received_monotonic,received_wall=self.latest;index=len(self.parts)
        path=self.args.output/f'grid_{index:04d}.npz';temporary=path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,occupancy=data,header_stamp_ns=np.int64(metadata['header_stamp_ns']),
                map_load_time_ns=np.int64(metadata['map_load_time_ns']),resolution_m=np.float64(metadata['resolution_m']),
                width=np.int64(metadata['width']),height=np.int64(metadata['height']),frame_id=np.asarray(metadata['frame_id']),
                origin_position_m=np.asarray(metadata['origin_position_m']),origin_quaternion_xyzw=np.asarray(metadata['origin_quaternion_xyzw']))
        os.replace(temporary,path)
        sidecar={'schema_version':1,'source':self.source,'grid':metadata,'snapshot_reason':reason,'elapsed_seconds':time.monotonic()-self.started,
                 'received_wall_time_ns':received_wall,'latest_received_age_seconds':time.monotonic()-received_monotonic,
                 'capture_sha256':sha(path),'bytes':path.stat().st_size}
        atomic_json(path.with_suffix('.json'),sidecar)
        self.parts.append({'name':path.name,'sha256':sidecar['capture_sha256'],'metadata_sha256':sha(path.with_suffix('.json')),
                           'bytes':sidecar['bytes'],'header_stamp_ns':metadata['header_stamp_ns'],'reception_sequence':metadata['reception_sequence'],'elapsed_seconds':sidecar['elapsed_seconds'],'reason':reason})
        self.last_saved_monotonic=time.monotonic()

    def monitor(self):
        if self.check_lifecycle() and self.latest and time.monotonic()-self.last_saved_monotonic>=self.args.interval:
            self.save_latest('periodic')

    def close_capture(self,reason):
        try:
            if self.latest and (not self.parts or self.parts[-1]['reception_sequence']!=self.latest[1]['reception_sequence']):self.save_latest('final')
            if not self.latest:self.failure=self.failure or 'no_valid_navigation_grid_received'
            if sha(__file__)!=self.source['script_sha256']:self.failure=self.failure or 'recorder_source_changed'
        except Exception as error:self.failure=self.failure or 'snapshot_write_failed:'+str(error)
        normal=reason in ('acquisition_finished','runtime_stopped')
        manifest={'schema_version':1,'capture_state':'CLOSED' if not self.failure and normal else 'INCOMPLETE',
                  'close_reason':self.failure or reason,'run_id':self.args.run_id,'runtime_generation':self.generation,'source':self.source,
                  'interval_seconds':self.args.interval,'elapsed_seconds':time.monotonic()-self.started,'started_wall_time_ns':self.started_wall_ns,
                  'statistics':dict(self.stats),'snapshots':self.parts,'total_bytes':sum(p['bytes'] for p in self.parts),
                  'latest_received_age_seconds':time.monotonic()-self.latest[2] if self.latest else None,
                  'final_latest_stamp_saved':bool(self.parts and self.latest and self.parts[-1]['reception_sequence']==self.latest[1]['reception_sequence']),
                  'meaning':'Passive snapshots of actual published navigation grid; CLOSED is a recording lifecycle result, not completeness or mapping acceptance.'}
        atomic_json(self.args.output/'capture_manifest.json',manifest);self.owner['state']=manifest['capture_state'];atomic_json(self.args.output/'owner.json',self.owner)
        print(json.dumps({'capture_state':manifest['capture_state'],'close_reason':manifest['close_reason'],'snapshots':len(self.parts),'output':str(self.args.output)},ensure_ascii=False),flush=True)
        return 0 if manifest['capture_state']=='CLOSED' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--duration',type=float,default=1900.);parser.add_argument('--interval',type=float,default=60.)
    parser.add_argument('--runtime-status',type=Path,default=Path('/workspace/deployment/runtime_status.json'))
    parser.add_argument('--run-status',type=Path,default=Path('/workspace/deployment/run_status.json'));args=parser.parse_args()
    if not args.run_id.strip():parser.error('run-id cannot be empty')
    if not math.isfinite(args.duration) or not 1<=args.duration<=7200:parser.error('duration must be 1..7200 seconds')
    if not math.isfinite(args.interval) or not 1<=args.interval<=600:parser.error('interval must be 1..600 seconds')
    args.output.mkdir(parents=True,exist_ok=False);stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);node=None;reason='duration_elapsed';code=1
    try:
        node=GridRecorder(args)
        while rclpy.ok() and not stopped[0] and not node.failure and not node.close_requested and time.monotonic()-node.started<args.duration:
            rclpy.spin_once(node,timeout_sec=.1);node.monitor()
        reason='operator_stopped' if stopped[0] else node.close_requested or reason
    except Exception as error:
        if node:node.failure='recorder_exception:'+type(error).__name__+':'+str(error)
        else:atomic_json(args.output/'capture_manifest.json',{'schema_version':1,'capture_state':'INCOMPLETE','close_reason':'startup_failed:'+str(error),'run_id':args.run_id,'snapshots':[]})
    finally:
        if node:code=node.close_capture(reason);node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return code


if __name__=='__main__':raise SystemExit(main())
