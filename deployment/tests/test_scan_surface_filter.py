"""Actual mapper admission: repeated ghosts, duplicate rays, planes and corners."""
import json,os,signal,struct,subprocess,time
import numpy as np
import pytest
from scipy.spatial import cKDTree
from test_temporal_confidence import MAPPER,no_core,read_native_points

@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('angle',[0.,.63])
def test_admission_rejects_persistent_ghosts_without_losing_planar_corner(tmp_path,enabled,angle):
 import rclpy
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import PointCloud2,PointField
 from std_srvs.srv import Trigger
 from rclpy.qos import qos_profile_sensor_data
 assert os.environ.get('ROS_DOMAIN_ID')=='178'
 namespace='/surface_'+str(int(angle*100))+('_on' if enabled else '_off');pcd=tmp_path/'native.pcd'
 settings={'lidar_topic':namespace+'/lidar','odom_topic':namespace+'/odom','map_topic':namespace+'/map','preview_topic':namespace+'/preview','filtered_lidar_topic':namespace+'/filtered','save_service':namespace+'/save','save_path':str(pcd),'scan_surface_filter_enabled':str(enabled).lower(),'voxel_size':'.025','native_confidence_filter_enabled':'true','native_temporal_support_enabled':'true','native_temporal_patch_enabled':'true','native_ground_temporal_filter_enabled':'true','native_rejected_evidence_enabled':'true','ground_snap_enabled':'true','stationary_mapping':'true','stationary_settle_seconds':'0.','publish_every_n_scans':'1','preview_every_n_scans':'1','checkpoint_every_n_scans':'0'}
 cmd=[str(MAPPER),'--ros-args']
 for k,v in settings.items():cmd+=['-p',k+':='+v]
 matrix=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
 plane=np.array([(4.,y,z) for y in np.arange(-.8,.81,.08) for z in np.arange(.55,1.76,.08)])
 other=np.array([(x,.8,z) for x in np.arange(3.2,4.01,.08) for z in np.arange(.55,1.76,.08)])
 ghosts=np.array([(5.5,.1,1.1),(5.53,.12,1.12),(4.14,.1,1.1)])
 ground=np.array([(2.,0.,0.)]);expected=np.concatenate([plane,other]);world=np.concatenate([plane,other,ghosts,ground])@matrix.T
 ghosts=ghosts@matrix.T;expected=expected@matrix.T
 local=world-np.array([0.,0.,.3]);local=np.concatenate([local,np.repeat((ghosts[0]-[0,0,.3])[None,:],150,axis=0)])
 with (tmp_path/'mapper.log').open('w') as log:
  process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=no_core)
  rclpy.init();node=rclpy.create_node('surface_injection');odom=node.create_publisher(Odometry,namespace+'/odom',qos_profile_sensor_data);lidar=node.create_publisher(PointCloud2,namespace+'/lidar',qos_profile_sensor_data);maps=[];previews=[];filtered=[]
  subs=[node.create_subscription(PointCloud2,namespace+'/map',maps.append,10),node.create_subscription(PointCloud2,namespace+'/preview',previews.append,10),node.create_subscription(PointCloud2,namespace+'/filtered',filtered.append,qos_profile_sensor_data)];save=node.create_client(Trigger,namespace+'/save')
  def wait(check,timeout=5):
   end=time.monotonic()+timeout
   while time.monotonic()<end:
    if check():return True
    rclpy.spin_once(node,timeout_sec=.02)
   return bool(check())
  try:
   assert wait(lambda:lidar.get_subscription_count() and odom.get_subscription_count() and save.service_is_ready())
   for stamp in [10_000_000,210_000_000,410_000_000]:
    o=Odometry();o.header.stamp.sec=1793000000;o.header.stamp.nanosec=stamp;o.pose.pose.orientation.w=1.
    for _ in range(4):odom.publish(o);wait(lambda:False,.03)
    msg=PointCloud2();msg.header.stamp=o.header.stamp;msg.header.frame_id='lidar';msg.width=len(local);msg.height=1;msg.point_step=12;msg.row_step=12*len(local);msg.fields=[PointField(name=n,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,n in enumerate('xyz')];msg.data=b''.join(struct.pack('<fff',*v) for v in local);before=len(maps);lidar.publish(msg);assert wait(lambda:len(maps)>before),(tmp_path/'mapper.log').read_text()
   assert wait(lambda:previews and len(filtered)>=3)
   response=save.call_async(Trigger.Request());assert wait(response.done);assert response.result().success
   pts=read_native_points(pcd);tree=cKDTree(pts);gd=tree.query(ghosts)[0]
   if enabled:assert np.all(gd>.04),gd
   else:assert np.all(gd<1e-5),gd
   # Both measured walls and their intersection remain supported within 5 cm.
   coverage=np.mean(tree.query(expected)[0]<.05);assert coverage>.98,coverage
   assert tree.query(ground@matrix.T)[0][0]<1e-5
   def points(msg):
    dtype=np.dtype(dict(names=list('xyz'),formats=['<f4']*3,offsets=[next(f.offset for f in msg.fields if f.name==n) for n in 'xyz'],itemsize=msg.point_step))
    a=np.ndarray((msg.height,msg.width),dtype=dtype,buffer=msg.data,strides=(msg.row_step,msg.point_step));return np.column_stack([a[n].ravel() for n in 'xyz'])
   for msg in [maps[-1],previews[-1]]:
    q=points(msg);assert len(q)==len(pts);assert cKDTree(q).query(pts)[0].max()<1e-5
   assert filtered[-1].header.stamp.nanosec==410_000_000
   if enabled:assert cKDTree(points(filtered[-1])+[0,0,.3]).query(ghosts)[0].min()>.04
   stats=json.loads((tmp_path/'native.pcd.acquisition.json').read_text());assert stats['scan_surface_filter_enabled']==enabled
   assert (stats['surface_rejected_measurements']>0)==enabled
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
   node.destroy_node()
   if rclpy.ok():rclpy.shutdown()
