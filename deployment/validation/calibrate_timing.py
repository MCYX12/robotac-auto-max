#!/usr/bin/env python3
"""Offline timing diagnosis from recorded raw ROS messages; no map rewriting."""
import argparse,json,sqlite3,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2
from map_audit import ROOT,boxes_from_xml

def load(path):
    cache=path.with_suffix('.npz')
    if cache.exists():return dict(np.load(cache))
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    topics={row[0]:(row[1],get_message(row[2])) for row in db.execute('select id,name,type from topics')}
    od=[];frames=[];clouds=[]
    for tid,payload in db.execute('select topic_id,data from messages order by timestamp'):
        name,typ=topics[tid]
        if name not in ('/front_lidar','/odom/mujoco_odom'):continue
        m=deserialize_message(payload,typ);t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if name=='/odom/mujoco_odom':
            v=m.pose.pose;od.append([t,v.position.x,v.position.y,v.position.z,v.orientation.x,v.orientation.y,v.orientation.z,v.orientation.w,m.twist.twist.angular.z])
        elif len(frames)%2==0:
            a=point_cloud2.read_points(m,field_names=('x','y','z'),skip_nans=True)
            xyz=np.column_stack([a[k] for k in ('x','y','z')]);xyz=xyz[np.isfinite(xyz).all(axis=1)]
            if len(xyz)>1:xyz=xyz[np.r_[True,np.any(np.diff(xyz,axis=0)!=0,axis=1)]]
            r=np.linalg.norm(xyz,axis=1);xyz=xyz[(r>.6)&(r<20)]
            if len(xyz)<200:continue
            index=np.linspace(0,len(xyz)-1,200,dtype=int);clouds.append(xyz[index]);frames.append(t)
        else:
            # Use every frame; fixed point budget keeps the diagnosis bounded.
            a=point_cloud2.read_points(m,field_names=('x','y','z'),skip_nans=True)
            xyz=np.column_stack([a[k] for k in ('x','y','z')]);xyz=xyz[np.isfinite(xyz).all(axis=1)]
            if len(xyz)>1:xyz=xyz[np.r_[True,np.any(np.diff(xyz,axis=0)!=0,axis=1)]]
            r=np.linalg.norm(xyz,axis=1);xyz=xyz[(r>.6)&(r<20)]
            if len(xyz)<200:continue
            index=np.linspace(0,len(xyz)-1,200,dtype=int);clouds.append(xyz[index]);frames.append(t)
    db.close();od=np.asarray(od);order=np.argsort(od[:,0]);od=od[order];od=od[np.r_[True,np.diff(od[:,0])>0]]
    result={'odom':od,'times':np.array(frames),'clouds':np.asarray(clouds)}
    np.savez_compressed(cache,**result);return result

def main(a):
    data=load(a.bag);odom=data['odom'];t=data['times'];raw=data['clouds']
    keep=(t>odom[0,0]+.2)&(t<odom[-1,0]-.1);t=t[keep];raw=raw[keep]
    base=raw+np.array([0,0,.3]);sl=Slerp(odom[:,0],Rotation.from_quat(odom[:,4:8]));boxes=boxes_from_xml(ROOT/'runtime/src/robot_mujoco/zsibot_robots/xgb/unreal.xml')
    def transformed(offset):
        times=t+offset;rot=sl(times);pos=np.column_stack([np.interp(times,odom[:,0],odom[:,k]) for k in (1,2,3)])
        xyz=np.einsum('nij,nkj->nki',rot.as_matrix(),base)+pos[:,None,:];return xyz
    initial=transformed(-.04);ground=np.abs(initial[:,:,2])<.12;walls=(initial[:,:,2]>.4)&(initial[:,:,2]<2.5)
    scores=[]
    for offset in np.arange(-.15,.051,.005):
        xyz=transformed(offset);wall=xyz[walls];best=np.full(len(wall),np.inf)
        for _,pos,size,rot in boxes:
            local=(wall[:,:2]-pos[:2])@rot;q=np.abs(local)-size[:2]
            sd=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0);best=np.minimum(best,np.abs(sd))
        floor=np.abs(xyz[:,:,2][ground]);scores.append({'offset':round(float(offset),4),'floor_median':float(np.median(floor)),'floor_p90':float(np.percentile(floor,90)),'wall_median':float(np.median(best)),'wall_p90':float(np.percentile(best,90)),'wall_p95':float(np.percentile(best,95))})
    result={'frames':len(t),'point_samples':int(raw.shape[0]*raw.shape[1]),'time_span_s':float(t[-1]-t[0]),'scores':scores,'best_floor':min(scores,key=lambda s:s['floor_p90']),'best_wall':min(scores,key=lambda s:s['wall_p90'])}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='scores'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('bag',type=Path);p.add_argument('--output',required=True,type=Path);main(p.parse_args())
