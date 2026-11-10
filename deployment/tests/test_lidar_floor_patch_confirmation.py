"""Short LiDAR bursts cannot borrow floor trust; stable sparse floor survives."""
import json,os,signal,struct,subprocess,time
import numpy as np
import pytest
from scipy.spatial import cKDTree
from test_temporal_confidence import MAPPER,no_core,read_native_points

@pytest.mark.parametrize('setting',[
 'native_lidar_ground_patch_minimum_frames:=2',
 'native_lidar_ground_patch_minimum_frames:=101',
 'native_lidar_ground_patch_minimum_span:=.nan',
 'native_lidar_ground_patch_minimum_span:=0.29',
 'native_lidar_ground_patch_minimum_span:=5.1',
])
def test_invalid_lidar_floor_patch_confirmation(setting):
 assert os.environ.get('ROS_DOMAIN_ID')=='178'
 result=subprocess.run([str(MAPPER),'--ros-args','-p',setting],capture_output=True,text=True,timeout=5,preexec_fn=no_core)
 assert result.returncode!=0 and 'Invalid mapper parameters' in result.stdout+result.stderr

@pytest.mark.parametrize('long_confirmation',[False,True])
def test_transient_floor_burst_and_long_sparse_measured_support(tmp_path,long_confirmation):
 import rclpy
 from nav_msgs.msg import Odometry
 from sensor_msgs.msg import PointCloud2,PointField
 from std_srvs.srv import Trigger
 from rclpy.qos import qos_profile_sensor_data
 assert os.environ.get('ROS_DOMAIN_ID')=='178'
 namespace='/floor_patch_long' if long_confirmation else '/floor_patch_legacy';pcd=tmp_path/'native.pcd'
 params={'lidar_topic':namespace+'/lidar','depth_ground_topic':namespace+'/depth','odom_topic':namespace+'/odom','map_topic':namespace+'/map','preview_topic':namespace+'/preview','save_service':namespace+'/save','save_path':str(pcd),'voxel_size':'.025','native_confidence_filter_enabled':'true','native_temporal_support_enabled':'true','native_temporal_patch_enabled':'true','native_ground_temporal_filter_enabled':'true','native_lidar_ground_patch_minimum_frames':'6' if long_confirmation else '3','native_lidar_ground_patch_minimum_span':'1.0' if long_confirmation else '.3','native_rejected_evidence_enabled':'true','ground_snap_enabled':'true','stationary_mapping':'true','stationary_settle_seconds':'0.','publish_every_n_scans':'1','preview_every_n_scans':'1','checkpoint_every_n_scans':'0'}
 cmd=[str(MAPPER),'--ros-args']
 for k,v in params.items():cmd+=['-p',k+':='+v]
 with (tmp_path/'mapper.log').open('w') as log:
  process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=no_core)
  rclpy.init();node=rclpy.create_node('lidar_floor_confirmation_injection');odom=node.create_publisher(Odometry,namespace+'/odom',qos_profile_sensor_data);lidar=node.create_publisher(PointCloud2,namespace+'/lidar',qos_profile_sensor_data);depth=node.create_publisher(PointCloud2,namespace+'/depth',qos_profile_sensor_data);maps=[];previews=[];subs=[node.create_subscription(PointCloud2,namespace+'/map',maps.append,10),node.create_subscription(PointCloud2,namespace+'/preview',previews.append,10)];save=node.create_client(Trigger,namespace+'/save')
  def wait(check,timeout=5):
   end=time.monotonic()+timeout
   while time.monotonic()<end:
    if check():return True
    rclpy.spin_once(node,timeout_sec=.02)
   return bool(check())
  def send(stamp,points,publisher):
   o=Odometry();o.header.stamp.sec=1793000000+stamp//10**9;o.header.stamp.nanosec=stamp%10**9;o.pose.pose.orientation.w=1.
   for _ in range(4):odom.publish(o);wait(lambda:False,.025)
   msg=PointCloud2();msg.header.stamp=o.header.stamp;msg.width=len(points);msg.height=1;msg.point_step=12;msg.row_step=12*len(points);msg.fields=[PointField(name=n,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,n in enumerate('xyz')];msg.data=b''.join(struct.pack('<fff',*v) for v in points);before=len(maps);publisher.publish(msg);assert wait(lambda:len(maps)>before),(tmp_path/'mapper.log').read_text()
  try:
   assert wait(lambda:lidar.get_subscription_count() and depth.get_subscription_count() and save.service_is_ready())
   transient=[(6.0125+i*.025,.0125,-.3) for i in range(4)]
   sparse=[(7.0125+i*.025,.0125,-.3) for i in range(4)]
   sparse_depth=[(9.0125+i*.025,.0125,-.3) for i in range(4)]
   direct=(8.0125,.0125,-.3);wall=(5.0125,.0125,.7)
   times=[10_000_000,110_000_000,210_000_000,310_000_000,610_000_000,810_000_000,1_010_000_000]
   for i,stamp in enumerate(times):
    pts=[sparse[i%4],direct,wall]+([transient[i]] if i<4 else [])
    send(stamp,pts,lidar)
    # Depth remains usable with the original 3-frame/0.3 s evidence.
    if i<4:send(stamp+1_000_000,[sparse_depth[i]],depth)
   future=save.call_async(Trigger.Request());assert wait(future.done);assert future.result().success
   points=read_native_points(pcd);tree=cKDTree(points)
   def contains(values):
    values=np.asarray(values)+[0,0,.3];return tree.query(values)[0]<1e-5
   assert np.all(contains(sparse));assert contains([direct])[0];assert np.all(contains(sparse_depth))
   assert np.all(contains(transient)==(not long_confirmation))
   stats=json.loads((tmp_path/'native.pcd.acquisition.json').read_text());assert stats['native_lidar_ground_patch_minimum_frames']==(6 if long_confirmation else 3)
   assert stats['native_lidar_ground_patch_minimum_span']==(1.0 if long_confirmation else .3)
   assert len(previews[-1].data)//previews[-1].point_step==len(points)
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
   node.destroy_node()
   if rclpy.ok():rclpy.shutdown()
