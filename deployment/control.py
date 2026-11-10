#!/usr/bin/env python3
"""One-command start, bounded cancellation and truthful status."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,re
from pathlib import Path
from datetime import datetime
from common import ROOT,STOP,read_json,atomic_json,runtime_problem,stop_group,process_alive,supervisor_alive,heartbeat_fresh,runtime_duration,disk_problem,result_lock

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','stop','status','recover']);parser.add_argument('run_id',nargs='?');args=parser.parse_args()
    if args.run_id and not re.fullmatch('[A-Za-z0-9_-]{1,80}',args.run_id):parser.error('编号只允许 1..80 个英文字母、数字、下划线或短横线')
    if args.action=='recover':
        # A new runner can acquire its lock before publishing STARTING. Avoid
        # replacing that new status using a stale record read by this command.
        recovery_lock=open(ROOT/'dev_ws/midterm/.portable_auto.lock','a')
        try:fcntl.flock(recovery_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return 0
        state=read_json(ROOT/'deployment/run_status.json')
        if state.get('state') in ('STARTING','RUNNING','SAVING','VALIDATING','STOPPING') and not supervisor_alive(state,str(ROOT/'deployment/run_session.py')):
            state.update(state='FAILED',reason='interrupted_supervisor_lost',heartbeat_monotonic=time.monotonic())
            location=state.get('output_directory')
            folder=Path(location).resolve() if isinstance(location,str) and location else None
            if folder and folder.is_dir() and folder.parent==(ROOT/'dev_ws/midterm/runs').resolve():
                with result_lock(folder):
                    atomic_json(folder/'result.json',state)
                    for f in folder.glob('*_motion.json'):
                        prior=read_json(f);prior.update(state='FAILED',reason='interrupted_supervisor_lost');atomic_json(f,prior)
            atomic_json(ROOT/'deployment/run_status.json',state)
        return 0
    if args.action=='status':
        for kind in ('runtime','run'):
            state=read_json(ROOT/f'deployment/{kind}_status.json')
            expected=str(ROOT/'deployment'/('simulation.py' if kind=='runtime' else 'run_session.py'))
            if state.get('state') in ('RUNNING','STARTING','SAVING','VALIDATING','STOPPING') and (not heartbeat_fresh(state) or not supervisor_alive(state,expected)):state.update(state='UNAVAILABLE',reason='supervisor_not_alive_or_stale')
            labels={'RUNNING':'运行中','STARTING':'启动中','SAVING':'保存中','VALIDATING':'检查地图质量中','NEEDS_REVIEW':'地图质量未通过','STOPPING':'停止中','COMPLETED':'采集及设定质量检查通过','FAILED':'失败','CANCELLED':'已取消','STOPPED':'已停止','UNAVAILABLE':'进程已退出'}
            print(('仿真' if kind=='runtime' else '最近任务')+'：'+labels.get(str(state.get('state')),'尚未启动'),flush=True)
            if state.get('run_id'):print('编号：'+str(state['run_id']),flush=True)
            if state.get('state') in ('FAILED','UNAVAILABLE','NEEDS_REVIEW'):print('原因：'+str(state.get('reason')),flush=True)
            if state.get('output_directory'):print('结果：'+str(state['output_directory']),flush=True)
        return 0
    if args.action=='stop':
        STOP.touch();print('已请求停止；正在保留本轮部分地图。',flush=True)
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            live=[]
            for name in ('runtime','run'):
                s=read_json(ROOT/f'deployment/{name}_status.json')
                expected=str(ROOT/'deployment'/('simulation.py' if name=='runtime' else 'run_session.py'))
                if supervisor_alive(s,expected):live.append(name)
            if not live:
                time.sleep(.4)
                return 0
            time.sleep(.3)
        print('停止等待超时，由 Docker 结束剩余进程。',flush=True);return 1
    try:runtime_duration(os.environ.get('ROBOTAC_RUNTIME_SECONDS','1800'))
    except ValueError as error:parser.error(str(error))
    problem=disk_problem()
    if problem:print('启动失败：'+problem,flush=True);return 2
    lock=open('/tmp/robotac_auto_max_start.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('已有完整任务，请勿重复启动。',flush=True);return 3
    existing=read_json(ROOT/'deployment/runtime_status.json')
    if supervisor_alive(existing,str(ROOT/'deployment/simulation.py')):
        print('仿真已运行或正在清理；请用 docker.sh run 或先停止再 start。',flush=True);return 3
    STOP.unlink(missing_ok=True);sim=None;mission=None;cancelled=[False];owns_runtime=False
    def signal_stop(*_):cancelled[0]=True;STOP.touch()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal_stop)
    try:
        sim=subprocess.Popen([sys.executable,str(ROOT/'deployment/simulation.py'),'--preserve-stop'],start_new_session=True)
        deadline=time.monotonic()+50
        while time.monotonic()<deadline:
            if cancelled[0]:return 130
            if sim.poll() is not None:raise RuntimeError('仿真启动失败，未开始控制')
            state=read_json(ROOT/'deployment/runtime_status.json')
            if state.get('supervisor_pid')==sim.pid:
                owns_runtime=True
                if not runtime_problem(state):break
            time.sleep(.2)
        else:raise RuntimeError('仿真启动超时')
        if cancelled[0] or STOP.exists():return 130
        runid=args.run_id or 'auto_max_'+datetime.now().strftime('%Y%m%d_%H%M%S')
        mission=subprocess.Popen(['bash',str(ROOT/'run_auto_max.sh'),runid],start_new_session=True)
        while mission.poll() is None:time.sleep(.2)
        return mission.returncode
    except Exception as e:print(str(e),file=sys.stderr,flush=True);return 2
    finally:
        if owns_runtime:STOP.touch()
        if mission and mission.poll() is None:
            try:mission.wait(timeout=35)
            except subprocess.TimeoutExpired:stop_group(mission)
        if sim:stop_group(sim,signal.SIGTERM)
if __name__=='__main__':raise SystemExit(main())
