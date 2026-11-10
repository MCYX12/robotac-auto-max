#!/usr/bin/env python3
"""Independent ROS observer: only subscribe, store bounded static LiDAR samples.
No publishers, services, SDK, control commands or map mutation.
"""
import argparse
from collections import defaultdict, deque
import json
import math
import os
from pathlib import Path
import queue
import signal
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from temporal_consistency import DEFAULTS, stamp_ns, odom_values, local_points, interpolate_pose, write_capture, sha


def read_state(path):
    try:
        value=json.loads(Path(path).read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):return {}


class Recorder(Node):
    def __init__(self,args):
        super().__init__('robotac_static_temporal_observer')
        self.args=args;self.params=dict(DEFAULTS);self.params['points_per_frame']=args.points_per_frame;self.started=time.monotonic();self.stats=defaultdict(int)
        self.buffer=deque();self.stable_since=-1;self.last_odom=-1;self.last_selected=-1
        self.odom_stamps=[];self.odom_receive=[];self.odom=[];self.frames=[];self.parts=[]
        self.frame_first_ns=None;self.frame_last_ns=None;self.odom_first_ns=None;self.odom_last_ns=None
        self.runtime_generation=read_state(args.runtime_status).get('generation');self.failure=None;self.close_requested=None;self.sequence=0;self.last_flush=time.monotonic();self.last_monitor=0.;self.maximum_callback_s=0.
        self.sources={'kind':'live_subscriber_samples','run_id':args.run_id,'recorder_script_sha256':sha(__file__),
                      'helper_script_sha256':sha(Path(__file__).with_name('temporal_consistency.py')),
                      'ros_domain_id':os.environ.get('ROS_DOMAIN_ID'),'ros_distro':os.environ.get('ROS_DISTRO')}
        self.sources['runtime_generation']=self.runtime_generation
        self.sources['cloud_topic']=args.cloud_topic
        self.sources['measurement_stage']='mapper_geometric_admission_before_temporal_accumulation' if args.cloud_topic=='/robotac/filtered_lidar' else 'raw_sensor_topic'
        self.owner={'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'run_id':args.run_id,'generation':self.runtime_generation,'script':str(Path(__file__).resolve()),'state':'RECORDING'}
        (args.output/'owner.json').write_text(json.dumps(self.owner,indent=2)+'\n')
        self.writer_queue=queue.Queue(maxsize=2);self.writer=threading.Thread(target=self.write_loop,name='static-sample-writer',daemon=True);self.writer.start()
        self.create_subscription(Odometry,'/odom/mujoco_odom',self.on_odom,qos_profile_sensor_data)
        self.create_subscription(PointCloud2,args.cloud_topic,self.on_cloud,qos_profile_sensor_data)

    def on_odom(self,msg):
        started=time.monotonic();self.stats['odom_received']+=1;t=stamp_ns(msg);v=np.asarray(odom_values(msg),np.float64)
        if t<=0 or not np.isfinite(v).all() or not .5<float(v[3:7]@v[3:7])<1.5:self.stats['invalid_odom']+=1;return
        if t<=self.last_odom:self.stats['nonincreasing_odom_stamp']+=1;return
        stable=np.linalg.norm(v[7:10])<=self.params['linear_limit'] and np.linalg.norm(v[10:13])<=self.params['angular_limit']
        if not stable:self.stable_since=-1
        elif self.stable_since<0 or (self.last_odom>0 and t-self.last_odom>100_000_000):self.stable_since=t
        self.last_odom=t;received=time.time_ns();self.buffer.append((t,v,self.stable_since))
        while len(self.buffer)>2 and t-self.buffer[0][0]>3_000_000_000:self.buffer.popleft()
        self.odom_stamps.append(t);self.odom_receive.append(received);self.odom.append(v)
        if self.odom_first_ns is None:self.odom_first_ns=t
        self.odom_last_ns=t;self.maximum_callback_s=max(self.maximum_callback_s,time.monotonic()-started)

    def on_cloud(self,msg):
        started=time.monotonic();self.stats['lidar_received']+=1;t=stamp_ns(msg)
        if t<=0:self.stats['invalid_lidar_stamp']+=1;return
        if self.last_selected>=0 and t-self.last_selected<round(1e9/self.params['sample_hz']):return
        self.last_selected=t;self.stats['lidar_rate_selected']+=1
        if len(self.buffer)<2:self.stats['outside_odom']+=1;return
        stamps=np.asarray([v[0] for v in self.buffer],np.int64);values=np.asarray([v[1] for v in self.buffer]);since=np.asarray([v[2] for v in self.buffer],np.int64)
        target=t+round(self.params['pose_offset_s']*1e9);matched,reason=interpolate_pose(stamps,values,since,target,self.params)
        if matched is None:self.stats[reason]+=1;return
        pose,bracket=matched
        try:xyz=local_points(msg,self.params['points_per_frame'],self.params['minimum_range_m'],self.params['maximum_range_m'])
        except (ValueError,KeyError,TypeError):self.stats['invalid_cloud']+=1;return
        if len(xyz)<100:self.stats['insufficient_local_points']+=1;return
        self.frames.append(dict(stamp=t,received=time.time_ns(),target=target,pose=pose,bracket=bracket,xyz=xyz));self.stats['stationary_frames']+=1
        if self.frame_first_ns is None:self.frame_first_ns=t
        self.frame_last_ns=t;self.maximum_callback_s=max(self.maximum_callback_s,time.monotonic()-started)

    def flush(self,final=False):
        if not self.odom and not self.frames:return
        task=(self.sequence,self.odom_stamps,self.odom_receive,self.odom,self.frames,dict(self.stats),dict(self.sources))
        try:self.writer_queue.put(task,block=final,timeout=15 if final else None)
        except queue.Full:self.failure='writer_queue_backlog';return
        self.sequence+=1;self.odom_stamps=[];self.odom_receive=[];self.odom=[];self.frames=[];self.last_flush=time.monotonic()

    def write_loop(self):
        while True:
            task=self.writer_queue.get()
            if task is None:self.writer_queue.task_done();return
            index,stamps,received,odom,frames,stats,source=task
            try:
                path=self.args.output/f'chunk_{index:04d}.npz'
                metadata=write_capture(path,stamps,received,odom,frames,self.params,stats,source)
                self.parts.append({'name':path.name,'sha256':metadata['capture_sha256'],'bytes':path.stat().st_size,'odom_samples':len(stamps),'stationary_frames':len(frames)})
            except Exception as error:self.failure='capture_write_failed:'+type(error).__name__+':'+str(error)
            finally:self.writer_queue.task_done()

    def monitor(self):
        now=time.monotonic()
        if now-self.last_monitor<.25:return
        self.last_monitor=now
        state=read_state(self.args.runtime_status);generation=state.get('generation')
        if generation and self.runtime_generation is None:
            self.runtime_generation=generation;self.sources['runtime_generation']=generation
            self.owner['generation']=generation;(self.args.output/'owner.json').write_text(json.dumps(self.owner,indent=2)+'\n')
        elif generation and generation!=self.runtime_generation:self.failure='runtime_generation_changed'
        if state.get('state')=='FAILED':self.failure='runtime_failed:'+str(state.get('reason'))
        elif state.get('state')=='STOPPED':self.close_requested='runtime_stopped'
        run=read_state(self.args.run_status)
        if run.get('run_id')==self.args.run_id:
            if run.get('state') in ('FAILED','CANCELLED'):self.failure='run_'+run['state'].lower()+':'+str(run.get('reason'))
            elif run.get('state') in ('SAVING','VALIDATING','COMPLETED','NEEDS_REVIEW'):self.close_requested='acquisition_finished'
        if time.monotonic()-self.last_flush>=60:self.flush()

    def close_capture(self,reason):
        self.flush(final=True)
        try:self.writer_queue.put(None,timeout=15)
        except queue.Full:self.failure=self.failure or 'writer_shutdown_queue_timeout'
        self.writer.join(timeout=30)
        if self.writer.is_alive():self.failure=self.failure or 'writer_shutdown_timeout'
        manifest={'schema_version':1,'capture_state':'INCOMPLETE' if self.failure else 'CLOSED',
                  'close_reason':self.failure or reason,'run_id':self.args.run_id,'runtime_generation':self.runtime_generation,
                  'parameters':self.params,'sources':self.sources,'statistics':dict(self.stats),'chunks':self.parts,
                  'elapsed_seconds':round(time.monotonic()-self.started,3),'maximum_callback_seconds':self.maximum_callback_s,
                  'first_frame_stamp_ns':self.frame_first_ns,'last_frame_stamp_ns':self.frame_last_ns,
                  'first_odom_stamp_ns':self.odom_first_ns,'last_odom_stamp_ns':self.odom_last_ns,
                  'total_bytes':sum(p['bytes'] for p in self.parts),
                  'meaning':'CLOSED means recording files were flushed, not complete mapping or a drift PASS. Analysis must establish comparable repeated observations.'}
        destination=self.args.output/'capture_manifest.json';temporary=destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,destination)
        self.owner['state']=manifest['capture_state'];(self.args.output/'owner.json').write_text(json.dumps(self.owner,indent=2)+'\n')
        print(json.dumps({'capture_state':manifest['capture_state'],'close_reason':manifest['close_reason'],'output':str(self.args.output),'chunks':len(self.parts),'statistics':dict(self.stats)},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--points-per-frame',type=int,choices=(3000,12000,20000),default=20000);
    parser.add_argument('--cloud-topic',choices=['/front_lidar','/robotac/filtered_lidar'],default='/front_lidar');
    parser.add_argument('--duration',type=float,default=1900.);parser.add_argument('--run-id',default='unspecified')
    parser.add_argument('--runtime-status',type=Path,default=Path('/workspace/deployment/runtime_status.json'))
    parser.add_argument('--run-status',type=Path,default=Path('/workspace/deployment/run_status.json'))
    args=parser.parse_args()
    if not math.isfinite(args.duration) or not 1<=args.duration<=7200:parser.error('duration must be 1..7200 seconds')
    args.output.mkdir(parents=True,exist_ok=False);stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);node=Recorder(args);reason='duration_elapsed'
    try:
        while rclpy.ok() and not stopped[0] and not node.failure and not node.close_requested and time.monotonic()-node.started<args.duration:
            rclpy.spin_once(node,timeout_sec=.1);node.monitor()
        if stopped[0]:reason='operator_stopped'
        elif node.close_requested:reason=node.close_requested
    except Exception as error:node.failure='recorder_exception:'+type(error).__name__+':'+str(error)
    finally:
        node.close_capture(reason);node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 1 if node.failure else 0


if __name__=='__main__':raise SystemExit(main())
