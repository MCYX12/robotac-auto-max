"""Isolated regression tests; publishers are replaced before control ticks."""
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from robotac_finals.exploration_controller import ExplorationController

@pytest.fixture
def controller():
    rclpy.init()
    n=ExplorationController()
    n.command_publisher=SimpleNamespace(publish=lambda msg:None)
    n._publish_status=lambda *a,**kw:None
    yield n
    n.destroy_node()
    if rclpy.ok():rclpy.shutdown()

def test_repeated_scan_pauses_cannot_hide_a_stuck_robot(controller):
    n=controller
    n.pose=(0.,0.,0.);n.goal=(5.,0.);n.path=[(0.,0.),(5.,0.)];n.path_index=1
    n.planned_goal=(5.,0.);n.plan_requested=False
    n.scan_pause_period=8.;n.scan_pause_duration=4.;n.next_scan_pause=108.
    n.progress_pose=n.pose;n.progress_time=100.;n.initial_panorama_done=True
    with patch('robotac_finals.exploration_controller.time.monotonic') as clock:
        for step in range(400):
            clock.return_value=100.+step*.1
            n.last_map_progress_time=clock.return_value
            n._control_tick()
            if n.recovery_count:break
    assert n.recovery_count==1
    assert 126. <= clock.return_value <= 127.

def test_scan_pause_freezes_existing_stuck_time(controller):
    n=controller;n.pose=(0.,0.,0.);n.progress_pose=(0.,0.,0.);n.progress_time=100.
    n._pause_progress(108.);n._pause_progress(111.);n._resume_progress(112.)
    assert n.progress_time==104.
    assert n.progress_pose==(0.,0.,0.)
    assert n.progress_pause_started is None

def test_real_progress_clears_old_pause(controller):
    n=controller;n.pose=(1.,0.,0.);n.progress_time=100.;n.progress_pause_started=108.
    with patch('robotac_finals.exploration_controller.time.monotonic', return_value=112.):n._reset_progress()
    assert n.progress_time==112. and n.progress_pose==n.pose and n.progress_pause_started is None

@pytest.mark.parametrize('field,value',[('position',float('nan')),('position',float('inf')),('quaternion',0.)])
def test_invalid_odom_cannot_replace_a_valid_pose(controller,field,value):
    n=controller;n.pose=(2.,3.,.4);msg=Odometry();msg.pose.pose.orientation.w=1.
    if field=='position':msg.pose.pose.position.x=value
    else:msg.pose.pose.orientation.w=value
    n._odom_callback(msg)
    assert n.pose==(2.,3.,.4)

def test_nearly_unit_quaternion_is_normalized(controller):
    msg=Odometry();msg.pose.pose.orientation.w=.9*math.cos(.2);msg.pose.pose.orientation.z=.9*math.sin(.2)
    controller._odom_callback(msg)
    assert controller.pose[2]==pytest.approx(.4)

def test_nonfinite_goal_cannot_replace_valid_goal(controller):
    controller.goal=(2.,3.);msg=PoseStamped();msg.pose.position.x=float('nan')
    controller._goal_callback(msg)
    assert controller.goal==(2.,3.)

@pytest.mark.parametrize('resolution,width,length',[(0.,2,4),(float('nan'),2,4),(.1,2,3)])
def test_malformed_grid_never_refreshes_map_freshness(controller,resolution,width,length):
    msg=OccupancyGrid();msg.info.width=width;msg.info.height=2;msg.info.resolution=resolution;msg.data=[0]*length
    controller._map_callback(msg)
    assert controller.last_map_progress_time is None and controller.map_message is None

def test_native_mapping_layout_and_continuity_boundaries(tmp_path):
    # Compile the exact dependency-free guard used by the deployed C++ mapper.
    compiler=shutil.which('g++')
    assert compiler, 'g++ is required for the mapper guard regression'
    root=Path(__file__).resolve().parents[2]
    header=root/'dev_ws/src/robotac_mapping_baseline/src/mapping_guard.hpp'
    cpp=tmp_path/'guards.cpp';binary=tmp_path/'guards'
    cpp.write_text(r'''
#include "mapping_guard.hpp"
#include <cassert>
#include <iostream>
using namespace robotac_mapping_baseline;
int main() {
 std::vector<CloudFieldLayout> f={{"x",0,7,1},{"y",4,7,1},{"z",8,7,1}};
 assert(cloud_layout_problem(2,2,12,32,64,false,f).empty()); // row padding supported
 assert(!cloud_layout_problem(0,2,12,32,64,false,f).empty());
 assert(!cloud_layout_problem(2,2,12,32,63,false,f).empty());
 assert(!cloud_layout_problem(2,2,12,20,40,false,f).empty());
 assert(!cloud_layout_problem(2,2,12,32,64,true,f).empty());
 f[0].datatype=8;assert(!cloud_layout_problem(2,2,12,32,64,false,f).empty());f[0].datatype=7;
 f[0].count=2;assert(!cloud_layout_problem(2,2,12,32,64,false,f).empty());f[0].count=1;
 f[0].offset=11;assert(!cloud_layout_problem(2,2,12,32,64,false,f).empty());f[0].offset=0;
 f[1].offset=2;assert(!cloud_layout_problem(2,2,12,32,64,false,f).empty());f[1].offset=4;
 f.push_back(f[0]);assert(!cloud_layout_problem(2,2,12,32,64,false,f).empty());
 auto check=[](double dt,double distance,double angle){return pose_continuity_problem(dt,distance,angle,.25,3.,.35,4.);};
 assert(check(.005,.265,.37).empty());
 assert(check(.005,.26501,.0)=="odometry_position_jump");
 assert(check(.005,.0,.37001)=="odometry_orientation_jump");
 assert(check(2.,.9,1.).empty()); // ordinary travel over dropped packets is allowed
 assert(check(.1,.05,.5).empty()); // standing/leg pitch with a broad margin
 assert(check(-.01,2.,2.).empty()); // bounded reordering discarded by caller
 assert(check(-.51,0.,0.)=="odometry_clock_reset");
 assert(check(.005,std::numeric_limits<double>::infinity(),0.)=="nonfinite_pose_difference");
 std::cout<<"18 mapper guard assertions passed\n";
}
''')
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-I',str(header.parent),str(cpp),'-o',str(binary)],check=True,capture_output=True,text=True)
    done=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
    assert '18 mapper guard assertions passed' in done.stdout

def test_mapper_process_rejects_bad_layout_and_preserves_prefault_checkpoint(tmp_path):
    """Actual native mapper, synthetic data, no control/SDK/simulation processes."""
    import os
    import resource
    import signal
    import struct
    from sensor_msgs.msg import PointCloud2, PointField
    from std_srvs.srv import Trigger
    from rclpy.qos import qos_profile_sensor_data
    root=Path(__file__).resolve().parents[2]
    executable=root/'dev_ws/install_humble/robotac_mapping_baseline/lib/robotac_mapping_baseline/odom_voxel_mapper'
    assert executable.is_file(), 'Build the updated mapper before this integration test'
    assert os.environ.get('ROS_DOMAIN_ID')=='178', 'Run isolated mapper injection in ROS_DOMAIN_ID=178'
    pcd=tmp_path/'synthetic.pcd';log_path=tmp_path/'mapper.log'
    def no_core():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    with log_path.open('w') as log:
        process=subprocess.Popen([str(executable),'--ros-args','-p','lidar_topic:=/guard/lidar',
            '-p','odom_topic:=/guard/odom','-p','map_topic:=/guard/map',
            '-p','preview_topic:=/guard/preview','-p','save_service:=/guard/save',
            '-p',f'save_path:={pcd}','-p','publish_every_n_scans:=0',
            '-p','preview_every_n_scans:=0','-p','checkpoint_every_n_scans:=0'],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=no_core)
        rclpy.init();node=rclpy.create_node('mapper_guard_injection')
        odom_pub=node.create_publisher(Odometry,'/guard/odom',qos_profile_sensor_data)
        cloud_pub=node.create_publisher(PointCloud2,'/guard/lidar',qos_profile_sensor_data)
        client=node.create_client(Trigger,'/guard/save')
        def spin_until(predicate,seconds=6.):
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                if predicate():return True
                rclpy.spin_once(node,timeout_sec=.03)
            return bool(predicate())
        def send_odom(nanosec,x=0.):
            msg=Odometry();msg.header.stamp.sec=100;msg.header.stamp.nanosec=nanosec
            msg.pose.pose.position.x=x;msg.pose.pose.orientation.w=1.
            odom_pub.publish(msg)
            for _ in range(3):rclpy.spin_once(node,timeout_sec=.02)
        def cloud(nanosec):
            msg=PointCloud2();msg.header.stamp.sec=100;msg.header.stamp.nanosec=nanosec
            msg.height=2;msg.width=2;msg.point_step=12;msg.row_step=32;msg.is_bigendian=False;msg.is_dense=True
            msg.fields=[PointField(name=n,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(('x','y','z'))]
            data=bytearray(64)
            for row in range(2):
                for col in range(2):struct.pack_into('<fff',data,row*32+col*12,1.+.1*(row*2+col),0.,0.)
            msg.data=bytes(data);return msg
        try:
            assert spin_until(lambda:odom_pub.get_subscription_count()>0 and cloud_pub.get_subscription_count()>0 and client.service_is_ready()),log_path.read_text()
            for i in range(1,6):
                stamp=i*10000000;send_odom(stamp);cloud_pub.publish(cloud(stamp))
                for _ in range(3):rclpy.spin_once(node,timeout_sec=.02)
            broken=cloud(60000000);broken.data=bytes(broken.data)[:-1]
            cloud_pub.publish(broken)
            cloud_pub.publish(cloud(50000000)) # duplicate timestamp must not add scan confidence
            for _ in range(5):rclpy.spin_once(node,timeout_sec=.02)
            assert process.poll() is None,log_path.read_text()
            future=client.call_async(Trigger.Request())
            assert spin_until(future.done),log_path.read_text()
            assert future.result().success
            before=json.loads(Path(str(pcd)+'.acquisition.json').read_text())
            assert before['invalid_cloud_frames']>=1
            assert before['nonadvancing_cloud_frames']>=1
            assert before['mapping_fault']==''
            # Same time base, 3m translation in 10ms. No corrupted point may enter.
            send_odom(60000000,3.)
            assert spin_until(lambda:process.poll() is not None),log_path.read_text()
            assert process.returncode != 0
            checkpoint=Path(str(pcd)+'.fault_checkpoint.pcd')
            assert checkpoint.is_file(),log_path.read_text()
            stats=json.loads(Path(str(checkpoint)+'.acquisition.json').read_text())
            assert stats['mapping_fault']=='odometry_position_jump'
            assert checkpoint.read_bytes()==pcd.read_bytes()
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3.)
            node.destroy_node()
            if rclpy.ok():rclpy.shutdown()
