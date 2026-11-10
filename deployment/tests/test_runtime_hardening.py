"""Regression tests for readiness, ownership and cancellation; no simulator/motion."""
import io
import fcntl
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from sensor_msgs.msg import PointCloud2, PointField, CompressedImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import control
import run_session
from telemetry import Telemetry


def bare_telemetry():
    node = object.__new__(Telemetry)
    node.counts = {k: 0 for k in ('lidar', 'odom', 'depth')}
    node.last = {k: 0. for k in node.counts}
    node.stamps = {}; node.errors = {}; node.pose = None; node.last_depth_decode = 0.
    return node


def cloud(xyz=(1., 2., 3.), *, endian=False, padded=False):
    msg = PointCloud2()
    msg.header.stamp.sec = 1
    msg.width = msg.height = 1
    msg.point_step = 12; msg.row_step = 16 if padded else 12
    msg.is_bigendian = endian
    msg.fields = [PointField(name=axis, offset=i*4, datatype=PointField.FLOAT32, count=1) for i, axis in enumerate('xyz')]
    msg.data = struct.pack(('>' if endian else '<')+'fff', *xyz) + (b'\0'*4 if padded else b'')
    return msg


@pytest.mark.parametrize('case', ('empty', 'missing_fields', 'bad_offset', 'all_nan', 'short_payload', 'short_row'))
def test_invalid_lidar_cannot_refresh_readiness(case):
    node = bare_telemetry(); msg = cloud()
    if case == 'empty':msg.data = b''; msg.row_step = 0
    elif case == 'missing_fields':msg.fields = []
    elif case == 'bad_offset':msg.fields[2].offset = 20
    elif case == 'all_nan':msg.data = struct.pack('<fff', *(float('nan'),)*3)
    elif case == 'short_payload':msg.data = b'123'
    elif case == 'short_row':msg.row_step = 8
    node.received('lidar', msg)
    assert node.counts['lidar'] == 0 and node.last['lidar'] == 0
    assert 'lidar' in node.errors


@pytest.mark.parametrize('endian,padded', ((False, False), (False, True), (True, False)))
def test_valid_lidar_layouts_are_accepted(endian, padded):
    node = bare_telemetry(); node.received('lidar', cloud(endian=endian, padded=padded))
    assert node.counts['lidar'] == 1 and not node.errors


@pytest.mark.parametrize('image', (np.zeros((480,640),np.uint16), np.zeros((480,640),np.uint8), np.zeros((479,640,3),np.uint8)))
def test_wrong_depth_encoding_cannot_refresh_readiness(image):
    node = bare_telemetry(); msg = CompressedImage(); msg.header.stamp.sec = 1
    msg.data = cv2.imencode('.png',image)[1].tobytes(); node.received('depth',msg)
    assert node.counts['depth'] == 0 and 'depth' in node.errors


def test_valid_depth_and_advanced_timestamps_required():
    node = bare_telemetry(); msg = CompressedImage(); msg.header.stamp.sec = 1
    msg.data = cv2.imencode('.png',np.zeros((480,640,4),np.uint8))[1].tobytes()
    node.received('depth',msg); node.received('depth',msg)
    assert node.counts['depth'] == 1
    msg.header.stamp.nanosec = 1_000_000_000
    node.received('depth',msg)
    assert node.counts['depth'] == 1 and 'depth' in node.errors


@pytest.mark.parametrize('value', ('null', '[]', '"text"', '{broken'))
def test_status_reader_rejects_non_object_json(tmp_path,value):
    path=tmp_path/'status.json';path.write_text(value)
    assert common.read_json(path)=={}


@pytest.mark.parametrize('pid', (0,-1,True,None,'1',1.5))
def test_process_identity_rejects_non_pid_values(pid):
    assert not common.process_alive(pid)


def test_supervisor_identity_rejects_reused_pid():
    pid=os.getpid(); command=Path('/proc/self/cmdline').read_bytes().split(b'\0')[0].decode()
    ticks=common.process_start_ticks(pid)
    assert common.supervisor_alive({'supervisor_pid':pid,'supervisor_start_ticks':ticks},command)
    assert not common.supervisor_alive({'supervisor_pid':pid,'supervisor_start_ticks':ticks+1},command)


@pytest.mark.parametrize('stamp', ('broken', None, float('nan'), float('inf'), -1))
def test_status_command_survives_invalid_heartbeat(tmp_path,monkeypatch,stamp,capsys):
    (tmp_path/'deployment').mkdir()
    (tmp_path/'deployment/run_status.json').write_text(json.dumps({'state':'RUNNING','heartbeat_monotonic':stamp,'supervisor_pid':os.getpid()}))
    monkeypatch.setattr(control,'ROOT',tmp_path);monkeypatch.setattr(sys,'argv',['control.py','status'])
    assert control.main()==0
    assert '进程已退出' in capsys.readouterr().out


@pytest.mark.parametrize('duration', ('nan','inf','-2','0','1801','bad'))
def test_bad_start_duration_rejected_before_launch(monkeypatch,duration):
    monkeypatch.setenv('ROBOTAC_RUNTIME_SECONDS',duration)
    monkeypatch.setattr(sys,'argv',['control.py','start','bad_duration'])
    monkeypatch.setattr(control.subprocess,'Popen',lambda *_a,**_k:pytest.fail('must not launch'))
    with pytest.raises(SystemExit) as error:control.main()
    assert error.value.code==2


def test_low_disk_rejected_before_launch(monkeypatch):
    monkeypatch.setattr(sys,'argv',['control.py','start','low_disk'])
    monkeypatch.setattr(control,'disk_problem',lambda:'insufficient_disk_space')
    monkeypatch.setattr(control.subprocess,'Popen',lambda *_a,**_k:pytest.fail('must not launch'))
    assert control.main()==2


def test_recovery_cannot_replace_a_new_runner_status(tmp_path,monkeypatch):
    (tmp_path/'deployment').mkdir();(tmp_path/'dev_ws/midterm').mkdir(parents=True)
    status=tmp_path/'deployment/run_status.json'
    status.write_text(json.dumps({'state':'STARTING','supervisor_pid':99999999}))
    with (tmp_path/'dev_ws/midterm/.portable_auto.lock').open('a') as runner:
        fcntl.flock(runner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        monkeypatch.setattr(control,'ROOT',tmp_path);monkeypatch.setattr(sys,'argv',['control.py','recover'])
        assert control.main()==0
        assert common.read_json(status)['state']=='STARTING'


@pytest.fixture
def session(tmp_path,monkeypatch):
    (tmp_path/'deployment').mkdir();(tmp_path/'dev_ws/midterm').mkdir(parents=True)
    monkeypatch.setattr(run_session,'ROOT',tmp_path)
    monkeypatch.setattr(run_session,'STOP',tmp_path/'deployment/stop.request')
    monkeypatch.setattr(run_session,'RUN_STATUS',tmp_path/'deployment/run_status.json')
    monkeypatch.setattr(run_session,'disk_problem',lambda *_a,**_k:'')
    monkeypatch.setattr(run_session,'runtime_problem',lambda:'')
    monkeypatch.setattr(run_session.signal,'signal',lambda *_a:None)
    monkeypatch.setattr(run_session.rclpy,'init',lambda **_k:None)
    monkeypatch.setattr(run_session.rclpy,'ok',lambda:False)
    monkeypatch.setattr(sys,'argv',['run_session.py','unit','--duration','30'])
    calls=[]; stopped=[]; capture=[]; verified=[]
    child=SimpleNamespace(pid=987654321,stdout=io.StringIO(''),poll=lambda:None)
    abort=SimpleNamespace(service_is_ready=lambda:True,call_async=lambda _r:calls.append('abort'))
    node=SimpleNamespace(create_client=lambda *_a:abort,destroy_node=lambda:None)
    monkeypatch.setattr(run_session.rclpy,'create_node',lambda *_a:node)
    monkeypatch.setattr(run_session,'stop_group',stopped.append)
    monkeypatch.setattr(run_session,'capture',lambda *_a:capture.append(True))
    monkeypatch.setattr(run_session,'verify',lambda *_a:verified.append(True))
    def popen(argv,**_kwargs):
        calls.append(argv)
        return child if argv[0]=='ros2' else SimpleNamespace(poll=lambda:0,returncode=0)
    monkeypatch.setattr(run_session.subprocess,'Popen',popen)
    folder=tmp_path/'dev_ws/midterm/runs/unit'
    return SimpleNamespace(root=tmp_path,folder=folder,stop=run_session.STOP,calls=calls,stopped=stopped,child=child,abort=abort,capture=capture,verified=verified,popen=popen)


def test_cancel_request_present_before_preflight_is_not_deleted(session):
    session.stop.touch()
    assert run_session.main()==130
    assert not session.calls and not session.capture and session.stop.exists()
    assert common.read_json(session.folder/'result.json')['state']=='CANCELLED'


def test_cancel_at_preflight_exit_prevents_launch(session,monkeypatch):
    def popen(argv,**_kwargs):
        session.calls.append(argv)
        def poll():session.stop.touch();return 0
        return SimpleNamespace(poll=poll,returncode=0)
    monkeypatch.setattr(run_session.subprocess,'Popen',popen)
    assert run_session.main()==130
    assert len(session.calls)==1 and not session.capture


def test_preflight_failure_keeps_exit_two_and_records_reason(session,monkeypatch):
    monkeypatch.setattr(run_session.subprocess,'Popen',lambda *_a,**_k:SimpleNamespace(poll=lambda:2,returncode=2))
    assert run_session.main()==2 and not session.capture
    result=common.read_json(session.folder/'result.json')
    assert result['state']=='FAILED' and result['reason']=='preflight_failed'


def test_cancel_waits_for_abort_service_instead_of_losing_request(session,monkeypatch):
    iterations=[0]
    def spin(*_a,**_k):
        iterations[0]+=1;session.stop.touch()
        if iterations[0]==3:
            common.atomic_json(session.folder/'unit_motion.json',{'state':'CANCELLED','reason':'operator_cancelled'})
    session.abort.service_is_ready=lambda:iterations[0]>=2
    monkeypatch.setattr(run_session.rclpy,'spin_once',spin)
    assert run_session.main()==130
    assert session.calls.count('abort')==1 and session.stopped==[session.child]
    assert common.read_json(session.folder/'result.json')['state']=='CANCELLED'


def test_low_disk_during_capture_preserves_failure_after_partial_abort(session,monkeypatch):
    iterations=[0]
    def spin(*_a,**_k):
        iterations[0]+=1
        if iterations[0]==2:
            (session.folder/'unit_native.pcd').write_bytes(b'partial')
            common.atomic_json(session.folder/'unit_motion.json',{'state':'CANCELLED','reason':'operator_cancelled'})
    monkeypatch.setattr(run_session.rclpy,'spin_once',spin)
    monkeypatch.setattr(run_session,'disk_problem',lambda *_a,**_k:'insufficient_disk_space' if iterations[0] else '')
    assert run_session.main()==4
    result=common.read_json(session.folder/'result.json')
    assert result['state']=='FAILED' and result['reason']=='insufficient_disk_space'
    assert result['partial_pcd'] and session.calls.count('abort')==1


def test_cancel_at_audit_exit_cannot_leave_completed_result(session,monkeypatch):
    def spin(*_a,**_k):
        (session.folder/'unit_native.pcd').write_bytes(b'native')
        common.atomic_json(session.folder/'unit_motion.json',{'state':'MAP_SAVED','saved_points':1000})
    monkeypatch.setattr(run_session.rclpy,'spin_once',spin)
    def popen(argv,**kwargs):
        if any(str(a).endswith('/audit_run.py') for a in argv):
            assert '--owner-pid' in argv and '--owner-start-ticks' in argv
            result=common.read_json(session.folder/'result.json');result['state']='COMPLETED'
            common.atomic_json(session.folder/'result.json',result);session.stop.touch()
        return session.popen(argv,**kwargs)
    monkeypatch.setattr(run_session.subprocess,'Popen',popen)
    assert run_session.main()==130
    assert session.verified and common.read_json(session.folder/'result.json')['state']=='CANCELLED'


def test_mapper_fault_checkpoint_is_reported_on_algorithm_failure(session,monkeypatch):
    checkpoint=session.folder/'unit_native.pcd.fault_checkpoint.pcd'
    monkeypatch.setattr(run_session.rclpy,'spin_once',lambda *_a,**_k:checkpoint.write_bytes(b'partial'))
    session.child.poll=lambda:1
    assert run_session.main()==4
    result=common.read_json(session.folder/'result.json')
    assert result['state']=='FAILED' and str(checkpoint) in result['checkpoint_files']


def test_observer_cancel_only_terminates_its_child(tmp_path):
    script=Path(__file__).resolve().parents[1]/'observer.py'
    token=f'{os.getpid()}-{time.time_ns()}'
    pidfile=tmp_path/'child.pid'
    child_code='import os,time,pathlib;pathlib.Path('+repr(str(pidfile))+').write_text(str(os.getpid()));time.sleep(60)'
    worker=subprocess.Popen([sys.executable,str(script),'run',token,sys.executable,'-c',child_code],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    try:
        deadline=time.monotonic()+5
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert pidfile.exists()
        assert subprocess.run([sys.executable,str(script),'cancel',token],timeout=5).returncode==0
        worker.communicate(timeout=12)
        assert worker.returncode==130 and unrelated.poll() is None
        assert not common.process_alive(int(pidfile.read_text()))
    finally:
        if worker.poll() is None:worker.kill();worker.communicate()
        unrelated.terminate();unrelated.wait(timeout=5)
