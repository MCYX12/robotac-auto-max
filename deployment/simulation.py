#!/usr/bin/env python3
"""Supervise the local vendor binaries and current sensor streams as one runtime."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from datetime import datetime
from pathlib import Path
import rclpy
from rclpy.signals import SignalHandlerOptions
from common import ROOT,STOP,atomic_json,read_json,process_alive,stop_group,process_start_ticks,disk_problem,MIN_SAVE_FREE_BYTES,supervisor_alive
from telemetry import Telemetry
from gpu_probe import GpuProbe
STATE=ROOT/'deployment/runtime_status.json'
FATAL=('Fatal error:','VK_ERROR_FEATURE_NOT_PRESENT','VK_ERROR_DEVICE_LOST','Signal 11 caught','Segmentation fault')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preserve-stop',action='store_true');args=parser.parse_args()
    lock=open('/tmp/robotac_auto_max_sim.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('仿真已运行，请勿重复启动。',flush=True);return 3
    problem=disk_problem()
    if problem:print('仿真启动失败：'+problem,flush=True);return 2
    if not args.preserve_stop:STOP.unlink(missing_ok=True)
    run_id=datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
    logs=ROOT/'deployment/logs'/run_id;logs.mkdir(parents=True)
    children={};handles={};offsets={};tails={};stop_requested=[False];stop_since=[None]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop_requested.__setitem__(0,True))
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);node=Telemetry();started=time.monotonic();ready=False;state='STARTING';reason='waiting_for_live_sensors';code=1
    gpu_probe=GpuProbe();last_write=0.;last_log_check=0.;last_loop=started;max_loop_delay=0.
    def write(force=False):
        nonlocal last_write
        now=time.monotonic()
        if not force and now-last_write<.25:return
        # Heartbeats are transient state: atomic replacement is needed, but
        # hundreds of fsyncs per second would stall the same loop they monitor.
        atomic_json(STATE,{'state':state,'reason':reason,'generation':run_id,'heartbeat_monotonic':now,'supervisor_pid':os.getpid(),'supervisor_start_ticks':process_start_ticks(os.getpid()),'process_pids':{k:c.pid for k,c in children.items()},'log_directory':str(logs),'headless':os.environ.get('ROBOTAC_HEADLESS','1')!='0','maximum_monitor_loop_delay_seconds':round(max_loop_delay,4),**node.summary()},durable=force)
        last_write=now
    def spawn(name,args,cwd,env=None):
        handles[name]=open(logs/(name+'.log'),'w');offsets[name]=0
        children[name]=subprocess.Popen(args,cwd=cwd,env=env,stdout=handles[name],stderr=subprocess.STDOUT,start_new_session=True)
        print(name,'PID',children[name].pid,flush=True)
    try:
        if STOP.exists():state='STOPPED';reason='cancelled_before_runtime_start';code=130;return code
        runtime=ROOT/'runtime';config=json.loads((runtime/'config/config.json').read_text())
        expected=json.loads((ROOT/'config/config.json').read_text())['robot']['sensors']['depth_sensor']
        if config['robot']['sensors']['depth_sensor']!=expected:raise RuntimeError('depth_calibration_mismatch')
        if config['robot']['mujoco_running'] is not True:raise RuntimeError('physics_disabled')
        sdk=(runtime/'src/robot_mc/build/export/config/sdk_config.yaml').read_text()
        if '127.0.0.1' not in sdk:raise RuntimeError('SDK_must_use_loopback')
        (runtime/'src/UeSim/Linux/UeSim/Content/model/config/config.json').write_text(json.dumps(config,indent=2)+'\n')
        # Disable optional visual effects implicated in the vendor Vulkan failure.
        # Geometry, collisions, LiDAR and calibrated depth dimensions stay unchanged.
        args=['./UeSim/Binaries/Linux/UeSim','UeSim','-game','-nosound','-noraytracing','-ResX=1280','-ResY=720','-ExecCmds=t.MaxFPS 30,r.DistanceFieldAO 0,r.AOGlobalDistanceField 0,r.DistanceFieldShadowing 0']
        if os.environ.get('ROBOTAC_HEADLESS','1')!='0':args.append('-RenderOffScreen')
        else:args.append('-windowed')
        spawn('ue',args,runtime/'src/UeSim/Linux');write(force=True)
        mc_started=False
        while True:
            rclpy.spin_once(node,timeout_sec=.1);now=time.monotonic()
            max_loop_delay=max(max_loop_delay,now-last_loop);last_loop=now
            gpu_ok=gpu_probe.tick(now)
            for name,child in children.items():
                if child.poll() is not None:raise RuntimeError(name+'_exited:'+str(child.returncode))
            if now-last_log_check>.25:
                last_log_check=now
                problem=disk_problem(ROOT,MIN_SAVE_FREE_BYTES)
                if problem:raise RuntimeError(problem)
                for name in children:
                    with (logs/(name+'.log')).open(errors='replace') as f:
                        f.seek(offsets[name]);chunk=f.read(65536);offsets[name]=f.tell()
                    text=tails.get(name,'')+chunk;tails[name]=text[-256:]
                    if any(x in text for x in FATAL):raise RuntimeError(name+'_fatal_log:'+next(x for x in FATAL if x in text))
            if not mc_started and now-started>=4:
                env=os.environ.copy();env.update(ROBOT_TYPE='XG',SDK_CLIENT_IP='127.0.0.1')
                spawn('controller',['./mc_ctrl','r'],runtime/'src/robot_mc/build/export/mc/bin',env)
                env=os.environ.copy();env['LIBGL_ALWAYS_SOFTWARE']='1'
                spawn('physics',['xvfb-run','-a','-s','-screen 0 1280x720x24 -nolisten tcp','./robot_mujoco'],runtime/'src/robot_mujoco/simulate/build',env)
                mc_started=True
            if not ready and node.ready() and gpu_ok:
                ready=True;state='RUNNING';reason='live_sensors_ready';print('仿真就绪：雷达、里程计、深度数据正常。',flush=True)
            if not ready and now-started>45:raise RuntimeError('startup_sensor_timeout:'+','.join(node.stale()))
            if ready and node.stale(5):raise RuntimeError('sensor_stream_stale:'+','.join(node.stale(5)))
            if stop_requested[0]:STOP.touch()
            if STOP.exists() or stop_requested[0]:
                if stop_since[0] is None:stop_since[0]=now
                state='STOPPING';reason='operator_stop_waiting_for_partial_save'
                run=read_json(ROOT/'deployment/run_status.json')
                if not supervisor_alive(run,str(ROOT/'deployment/run_session.py')) or run.get('state') not in ('STARTING','RUNNING','SAVING','VALIDATING','STOPPING') or now-stop_since[0]>35:
                    state='STOPPED';reason='operator_stopped';code=0;break
            write()
    except Exception as e:
        state='FAILED';reason=str(e);print('仿真失败：'+reason+'；日志：'+str(logs),file=sys.stderr,flush=True);write(force=True)
    finally:
        gpu_probe.close()
        for child in reversed(list(children.values())):stop_group(child,signal.SIGTERM)
        for handle in handles.values():handle.close()
        write(force=True);node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return code
if __name__=='__main__':raise SystemExit(main())
