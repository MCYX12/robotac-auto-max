import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import pytest,rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2,CompressedImage
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import runtime_problem
from robotac_finals.task1_mission_manager import Task1MissionManager
from robotac_finals.sdk_twist_bridge import SdkTwistBridge
from robotac_finals.exploration_controller import ExplorationController
from robotac_finals.frontier_selector import FrontierSelector

@pytest.fixture(autouse=True)
def ros_context():
    rclpy.init();yield
    if rclpy.ok():rclpy.shutdown()

@pytest.fixture
def manager(tmp_path):
    pcd=tmp_path/'native.pcd'
    pcd.write_text('VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1000\nHEIGHT 1\nPOINTS 1000\nDATA ascii\n'+'0 0 0\n'*1000)
    node=Task1MissionManager(parameter_overrides=[Parameter('output_pcd',value=str(pcd)),Parameter('status_path',value=str(tmp_path/'status.json')),Parameter('startup_sensor_timeout',value=0.),Parameter('save_retry_limit',value=1)])
    future=SimpleNamespace(done=lambda:True,result=lambda:SimpleNamespace(success=True,message='native saved'))
    node.save_client=SimpleNamespace(wait_for_service=lambda timeout_sec:True,call_async=lambda request:future)
    node.sensor_last={k:time.monotonic() for k in node.sensor_last}
    yield node
    node.destroy_node()

def status(node):return json.loads(node.status_path.read_text())

def test_healthy_save_is_pending_quality_not_completed(manager):
    manager._request_save('hard_deadline');manager._poll_save_future(time.monotonic())
    assert status(manager)['state']=='MAP_SAVED'
    assert not status(manager)['partial_pcd_saved']

def test_stale_sensor_failure_saves_partial_without_success(manager):
    manager.sensor_last['lidar']-=20
    manager._tick()
    assert status(manager)['state']=='FAILED'
    assert status(manager)['reason']=='sensor_stream_stale:lidar'
    assert status(manager)['partial_pcd_saved']

def test_backend_disappears_during_save_cannot_complete(manager,tmp_path):
    manager._request_save('hard_deadline')
    manager.runtime_health_required=True;manager.runtime_health_path=tmp_path/'missing.json'
    manager._poll_save_future(time.monotonic())
    assert status(manager)['state']=='FAILED'
    assert status(manager)['reason']=='runtime_health_missing'

def test_operator_cancellation_is_distinct_from_completion(manager):
    response=manager._abort_callback(None,SimpleNamespace())
    assert response.success
    manager._poll_save_future(time.monotonic())
    assert status(manager)['state']=='CANCELLED'
    assert status(manager)['partial_pcd_saved']

def test_empty_save_is_rejected_even_with_success_response(manager):
    manager.output_pcd.write_bytes(b'')
    manager._request_save('hard_deadline');manager._poll_save_future(time.monotonic())
    assert status(manager)['state']=='FAILED'
    assert status(manager)['reason']=='saved_pcd_empty'

def test_invalid_odometry_does_not_refresh_sensor_health(manager):
    old=manager.sensor_last['odom'];msg=Odometry();msg.pose.pose.position.x=float('nan')
    manager._sensor_callback('odom',msg)
    assert manager.sensor_last['odom']==old

def test_empty_depth_and_lidar_do_not_refresh_health(manager):
    old=dict(manager.sensor_last)
    manager._sensor_callback('depth',CompressedImage());manager._sensor_callback('lidar',PointCloud2())
    assert manager.sensor_last==old

def test_standing_failure_has_bounded_terminal_failure(manager):
    manager.control_wait_since=time.monotonic()-40
    manager._tick()
    assert status(manager)['state']=='FAILED'
    assert status(manager)['reason']=='standing_or_posture_failed_30s'

def test_missing_sensor_never_attempts_stand():
    n=SdkTwistBridge()
    class SDK:
        stands=0
        def getCurrentCtrlmode(self):return 0
        def standUp(self):self.stands+=1;return 0
    n.sdk=SDK()
    n._control_tick()
    assert n.sdk.stands==0
    assert n.last_state=='SENSOR_STALE_STOP'
    n.destroy_node()

def test_stale_runtime_heartbeat_rejected():
    assert runtime_problem({'state':'RUNNING','heartbeat_monotonic':time.monotonic()-10,'supervisor_pid':1})=='runtime_heartbeat_stale'


def test_runtime_generation_switch_cannot_mix_two_worlds(manager,tmp_path):
    manager.runtime_health_required=True;manager.runtime_generation='old'
    manager.runtime_health_path=tmp_path/'runtime.json'
    manager.runtime_health_path.write_text(json.dumps({'state':'RUNNING','generation':'new','heartbeat_monotonic':time.monotonic()}))
    manager._tick()
    assert status(manager)['state']=='FAILED'
    assert status(manager)['reason']=='runtime_generation_changed'

def test_nonfinite_runtime_heartbeat_rejected():
    assert runtime_problem({'state':'RUNNING','heartbeat_monotonic':float('nan'),'supervisor_pid':1})=='invalid_runtime_heartbeat'

def test_save_response_timeout_is_terminal_and_bounded(manager):
    manager.save_client.call_async=lambda _:SimpleNamespace(done=lambda:False,cancel=lambda:None)
    manager._request_save('hard_deadline');manager.save_future_deadline=time.monotonic()-1
    manager._tick()
    assert status(manager)['state']=='FAILED'
    assert status(manager)['reason']=='save_response_timeout'

def test_bad_quaternion_is_not_upright():
    n=SdkTwistBridge();msg=Odometry();msg.pose.pose.position.z=.3;msg.pose.pose.orientation.w=100.
    n._odom_callback(msg)
    assert not n.upright and n.last_odom==0.
    n.destroy_node()


def test_recover_dead_supervisor_marks_incomplete_run_failed(tmp_path,monkeypatch):
    import control
    root=tmp_path/'project';folder=root/'dev_ws/midterm/runs/interrupted';folder.mkdir(parents=True)
    (root/'deployment').mkdir()
    pending={'state':'RUNNING','supervisor_pid':99999999,'output_directory':str(folder)}
    (root/'deployment/run_status.json').write_text(json.dumps(pending))
    (folder/'interrupted_motion.json').write_text(json.dumps({'state':'RUNNING'}))
    monkeypatch.setattr(control,'ROOT',root);monkeypatch.setattr(sys,'argv',['control.py','recover'])
    assert control.main()==0
    assert json.loads((folder/'result.json').read_text())['state']=='FAILED'
    assert json.loads((folder/'interrupted_motion.json').read_text())['reason']=='interrupted_supervisor_lost'

def test_stationary_scan_pause_commands_zero_without_erasing_stuck_history():
    node=ExplorationController()
    try:
        node.pose=(0.,0.,0.);node.goal=(5.,0.)
        node.last_map_progress_time=time.monotonic()
        node.scan_pause_period=8.;node.next_scan_pause=0.
        commands=[];states=[]
        node.command_publisher=SimpleNamespace(publish=commands.append)
        node._publish_status=lambda state,**kw:states.append(state)
        node.progress_time=0.;node.progress_pose=(0.,0.)
        node._control_tick()
        assert commands and commands[-1].linear.x==0 and commands[-1].angular.z==0
        assert states[-1]=='SCANNING_STATIONARY'
        assert node.progress_time==0. and node.progress_pose==(0.,0.)
        assert node.progress_pause_started is not None
        node._resume_progress(node.progress_pause_started+4.)
        assert node.progress_time==4. and node.progress_pause_started is None
    finally:node.destroy_node()

def test_stale_navigation_map_takes_priority_over_scan_pause():
    node=ExplorationController()
    try:
        node.pose=(0.,0.,0.);node.goal=(5.,0.)
        node.last_map_progress_time=time.monotonic()-20
        node.scan_pause_period=8.;node.next_scan_pause=0.
        commands=[];states=[]
        node.command_publisher=SimpleNamespace(publish=commands.append)
        node._publish_status=lambda state,**kw:states.append(state)
        node._control_tick()
        assert commands[-1].linear.x==0 and states[-1]=='NAV_MAP_STALE'
        assert node.scan_pause_until==0
    finally:node.destroy_node()

def test_initial_panorama_waits_for_sdk_safety_before_turning():
    node=ExplorationController()
    try:
        node.initial_panorama_done=False;node.pose=(0.,0.,0.)
        node.safety_state='SENSOR_STALE_STOP';commands=[]
        node.command_publisher=SimpleNamespace(publish=commands.append)
        node._control_tick()
        assert commands[-1].angular.z==0 and commands[-1].linear.x==0
        assert node.panorama_started is None
    finally:node.destroy_node()

def test_initial_panorama_finishes_across_yaw_wrap_without_translation():
    import math
    node=ExplorationController()
    try:
        node.initial_panorama_done=False;node.safety_state='ACTIVE';commands=[]
        node.command_publisher=SimpleNamespace(publish=commands.append)
        for step in range(65):
            angle=step*.1;node.pose=(0.,0.,math.atan2(math.sin(angle),math.cos(angle)))
            node._control_tick()
        assert node.initial_panorama_done and node.panorama_angle>=2*math.pi-.05
        assert all(command.linear.x==0 for command in commands)
        assert commands[-1].angular.z==0
    finally:node.destroy_node()

def test_floor_priority_cannot_starve_early_frontier_exploration():
    node=FrontierSelector()
    try:
        node.frontiers_before_floor=3;node.frontiers_since_floor=10
        node.floor_priority_after=600;node.selection_started_at=time.monotonic()-100
        assert not node._floor_turn_due()
        node.selection_started_at=time.monotonic()-610
        assert node._floor_turn_due()
    finally:node.destroy_node()

@pytest.mark.parametrize('metadata_first',[False,True])
@pytest.mark.parametrize('goal_type',['FLOOR_REOBSERVE','STRUCTURE_REOBSERVE'])
def test_only_matching_observation_goal_aligns_in_both_topic_orders(metadata_first,goal_type):
    node=ExplorationController()
    try:
        node.align_goal_orientation=True;node.align_only_observation_goals=True
        goal=PoseStamped();goal.pose.position.x=2.;goal.pose.position.y=1.;goal.pose.orientation.z=.6;goal.pose.orientation.w=.8
        msg=String(data=json.dumps({'state':'GOAL_SELECTED','goal_type':goal_type,'x':2.,'y':1.,'observation_yaw':.7}))
        if metadata_first:node._goal_status_callback(msg)
        node._goal_callback(goal)
        if not metadata_first:node._goal_status_callback(msg)
        assert node.goal_yaw==.7
        node._goal_status_callback(String(data=json.dumps({'state':'GOAL_SELECTED','goal_type':'FRONTIER','x':2.,'y':1.,'observation_yaw':1.8})))
        assert node.goal_yaw is None
    finally:node.destroy_node()

def test_one_real_floor_point_per_cell_is_covered_not_a_density_hole():
    node=FrontierSelector()
    try:
        msg=OccupancyGrid();msg.info.resolution=.1;msg.header.frame_id='world'
        rows,cols=np.mgrid[:10,:10];xy=np.column_stack([cols.ravel(),rows.ravel()])*.1+.05
        node.floor_xy=xy;node.floor_frame='world';node.floor_stamp=node.get_clock().now().nanoseconds/1e9
        assert node._floor_gaps(msg,np.zeros((10,10),dtype=np.int8))==[]
        keep=~((rows>=2)&(rows<8)&(cols>=2)&(cols<8));node.floor_xy=xy[keep.ravel()]
        assert node._floor_gaps(msg,np.zeros((10,10),dtype=np.int8))
    finally:node.destroy_node()

def test_large_budget_allows_remote_floor_hole_but_still_has_a_bound():
    node=FrontierSelector()
    try:
        node.floor_detour_budget=600.;node.floor_maximum_path=60.;node.floor_reserved_seconds=0.
        assert node._floor_within_budget(30.)
        node.floor_reserved_seconds=500.
        assert not node._floor_within_budget(30.)
        assert not node._floor_within_budget(float('inf'))
    finally:node.destroy_node()

def test_transient_heartbeat_does_not_fsync_but_terminal_state_does(tmp_path,monkeypatch):
    import common
    calls=[];monkeypatch.setattr(common.os,'fsync',calls.append)
    common.atomic_json(tmp_path/'heartbeat.json',{'state':'RUNNING'},durable=False)
    assert json.loads((tmp_path/'heartbeat.json').read_text())['state']=='RUNNING' and not calls
    common.atomic_json(tmp_path/'terminal.json',{'state':'FAILED'})
    assert len(calls)==1

def test_gpu_probe_never_waits_for_a_pending_query(monkeypatch):
    import gpu_probe
    class Pending:
        def poll(self):return None
        def communicate(self,**kwargs):raise AssertionError('must not wait for pending query')
    monkeypatch.setattr(gpu_probe.subprocess,'Popen',lambda *a,**k:Pending())
    probe=gpu_probe.GpuProbe()
    assert not probe.tick(10.)
    assert not probe.tick(11.)

def test_gpu_probe_timeout_kills_only_its_query(monkeypatch):
    import gpu_probe
    class Query:
        killed=False
        def poll(self):return None
        def kill(self):self.killed=True
        def communicate(self,**kwargs):return ('','')
    child=Query();monkeypatch.setattr(gpu_probe.subprocess,'Popen',lambda *a,**k:child)
    probe=gpu_probe.GpuProbe();probe.tick(10.)
    with pytest.raises(RuntimeError,match='GPU_probe_timeout'):probe.tick(12.1)
    assert child.killed and probe.process is None

@pytest.mark.parametrize('code,output,error',[(0,'RTX 5070\n',''),(255,'','NVML failed')])
def test_gpu_probe_reports_actual_result(monkeypatch,code,output,error):
    import gpu_probe
    class Query:
        def poll(self):return code
        def communicate(self,**kwargs):return (output,error)
    monkeypatch.setattr(gpu_probe.subprocess,'Popen',lambda *a,**k:Query())
    probe=gpu_probe.GpuProbe();assert not probe.tick(10.)
    if code:
        with pytest.raises(RuntimeError,match='GPU_access_lost'):probe.tick(10.1)
    else:assert probe.tick(10.1)

def wall_fixture(node):
    msg=OccupancyGrid();msg.info.resolution=.1;msg.info.width=50;msg.info.height=50;msg.header.frame_id='world'
    grid=np.full((50,50),-1,dtype=np.int8);grid[6:45,3:47]=0;grid[5,3:47]=100;msg.data=grid.ravel().tolist()
    x,z=np.meshgrid(np.arange(.35,4.65,.05),np.arange(.3,2.71,.05))
    wall=np.column_stack([x.ravel(),np.full(x.size,.55),z.ravel()])
    node.structure_xyz=wall;node.structure_reobserve_enabled=True;node.floor_frame='world';node.floor_stamp=node.get_clock().now().nanoseconds/1e9
    return msg,grid,wall

def test_vertical_wall_deficit_is_detected_from_live_data_only():
    node=FrontierSelector()
    try:
        msg,grid,wall=wall_fixture(node)
        assert node._structure_gaps(msg,grid)==[]
        node.structure_xyz=wall[~((wall[:,0]>1.5)&(wall[:,0]<3.2)&(wall[:,2]>1.2))]
        clusters=node._structure_gaps(msg,grid)
        assert clusters and any(15<c[:,1].mean()<32 for c in clusters)
        node.floor_stamp-=50
        assert node._structure_gaps(msg,grid)==[]
    finally:node.destroy_node()

def test_wall_deficit_produces_safe_directed_observation_goal():
    node=FrontierSelector()
    try:
        msg,grid,wall=wall_fixture(node)
        node.structure_xyz=wall[~((wall[:,0]>1.5)&(wall[:,0]<3.2)&(wall[:,2]>1.2))]
        cells=np.argwhere(grid==0);node.floor_xy=(cells[:,::-1]+.5)*.1
        node.map_message=msg;node.robot_xy=np.array([2.5,3.2]);node.minimum_cluster=100000
        node.floor_viewpoint_enabled=True;node.floor_detour_budget=900.;node.floor_maximum_path=60.;node.path_cost_enabled=True
        goals=[];states=[];node.goal_publisher=SimpleNamespace(publish=goals.append);node.status_publisher=SimpleNamespace(publish=states.append)
        node._select()
        assert goals and json.loads(states[-1].data)['goal_type']=='STRUCTURE_REOBSERVE'
        pose=goals[-1].pose;cell=(int(pose.position.y/.1),int(pose.position.x/.1))
        assert grid[cell]==0 and np.isfinite(json.loads(states[-1].data)['observation_yaw'])
        assert node.floor_reserved_seconds>0
    finally:node.destroy_node()

def test_small_measured_floor_holes_are_not_ignored():
    node=FrontierSelector()
    try:
        msg=OccupancyGrid();msg.info.resolution=.1;msg.header.frame_id='world'
        rows,cols=np.mgrid[:10,:10];xy=np.column_stack([cols.ravel(),rows.ravel()])*.1+.05
        keep=~((rows>=3)&(rows<7)&(cols>=3)&(cols<6))
        node.floor_xy=xy[keep.ravel()];node.floor_frame='world';node.floor_stamp=node.get_clock().now().nanoseconds/1e9
        groups=node._floor_gaps(msg,np.zeros((10,10),dtype=np.int8))
        assert groups and any(2<=len(group)<6 for group in groups)
    finally:node.destroy_node()
