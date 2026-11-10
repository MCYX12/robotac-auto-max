"""Fresh, nonempty and calibrated sensor evidence; a process PID is not readiness."""
import math,time
import cv2,numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2,CompressedImage,PointField
from nav_msgs.msg import Odometry
class Telemetry(Node):
    def __init__(self,name='robotac_runtime_telemetry'):
        super().__init__(name)
        self.counts={k:0 for k in ('lidar','odom','depth')};self.last={k:0. for k in self.counts}
        self.stamps={};self.errors={};self.pose=None;self.last_depth_decode=0.
        self.subs=[self.create_subscription(t,topic,lambda msg,k=k:self.received(k,msg),qos_profile_sensor_data) for k,t,topic in [('lidar',PointCloud2,'/front_lidar'),('odom',Odometry,'/odom/mujoco_odom'),('depth',CompressedImage,'/front_depth/image/compressed')]]
    def received(self,k,msg):
        now=time.monotonic();stamp=(msg.header.stamp.sec,msg.header.stamp.nanosec)
        if stamp[0]<0 or not 0<=stamp[1]<1_000_000_000 or stamp<=(0,0):
            self.errors[k]='invalid sensor timestamp';return
        if k in self.stamps and stamp<=self.stamps[k]:return
        try:
            if k=='lidar':
                if msg.width*msg.height<1 or msg.point_step<12 or msg.row_step<msg.width*msg.point_step or len(msg.data)<msg.row_step*msg.height:raise ValueError('empty/malformed cloud')
                fields={f.name:f for f in msg.fields}
                formats=[];offsets=[]
                for axis in 'xyz':
                    field=fields.get(axis)
                    if field is None or field.count!=1 or field.datatype not in (PointField.FLOAT32,PointField.FLOAT64):raise ValueError('missing/invalid XYZ fields')
                    size=4 if field.datatype==PointField.FLOAT32 else 8
                    if field.offset<0 or field.offset+size>msg.point_step:raise ValueError('invalid XYZ field offset')
                    formats.append(('>' if msg.is_bigendian else '<')+f'f{size}');offsets.append(field.offset)
                dtype=np.dtype({'names':list('xyz'),'formats':formats,'offsets':offsets,'itemsize':msg.point_step})
                points=np.ndarray((msg.height,msg.width),dtype=dtype,buffer=msg.data,strides=(msg.row_step,msg.point_step))
                if not np.logical_and.reduce([np.isfinite(points[axis]) for axis in 'xyz']).any():raise ValueError('cloud has no finite XYZ point')
            elif k=='odom':
                p=msg.pose.pose;values=[p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]
                if not all(math.isfinite(v) for v in values) or not .5<sum(v*v for v in values[3:])<1.5:raise ValueError('invalid pose')
                self.pose=values
            elif now-self.last_depth_decode>=1.:
                img=cv2.imdecode(np.frombuffer(bytes(msg.data),np.uint8),cv2.IMREAD_UNCHANGED)
                if img is None or img.shape[:2]!=(480,640) or img.dtype!=np.uint8 or img.ndim!=3 or img.shape[2] not in (3,4):raise ValueError('invalid encoded BGR/BGRA depth image/calibration')
                self.last_depth_decode=now
            elif not msg.data:raise ValueError('empty depth frame')
        except (ValueError,TypeError,cv2.error) as e:
            self.errors[k]=str(e);return
        self.stamps[k]=stamp;self.counts[k]+=1;self.last[k]=now;self.errors.pop(k,None)
    def stale(self,limit=3.):
        now=time.monotonic();return [k for k,t in self.last.items() if not t or now-t>limit]
    def ready(self):return all(v>=3 for v in self.counts.values()) and not self.stale(1.5)
    def summary(self):
        now=time.monotonic();return {'sensor_counts':dict(self.counts),'sensor_age_seconds':{k:round(now-t,3) if t else None for k,t in self.last.items()},'invalid_sensor_messages':dict(self.errors),'pose_xyz_quaternion':self.pose}
