#!/usr/bin/env python3
"""Own one mission, save partial results on failure/cancellation, validate before success."""
import argparse,fcntl,hashlib,json,math,os,re,signal,subprocess,sys,threading,time
from pathlib import Path
import rclpy
from rclpy.signals import SignalHandlerOptions
from std_srvs.srv import Trigger
from common import ROOT,STOP,atomic_json,read_json,runtime_problem,stop_group,result_lock,process_start_ticks,disk_problem,MIN_SAVE_FREE_BYTES
from provenance import capture,verify
RUN_STATUS=ROOT/'deployment/run_status.json'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_id');parser.add_argument('--speed',type=float,default=.45);parser.add_argument('--duration',type=float,default=1800.)
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}',args.run_id):parser.error('编号只允许 1..80 个英文字母、数字、下划线或短横线')
    if not math.isfinite(args.speed) or not 0<args.speed<=.45:parser.error('speed must be in (0,.45]')
    if not math.isfinite(args.duration) or not 30<=args.duration<=1800:parser.error('duration must be 30..1800 seconds')
    lock=open(ROOT/'dev_ws/midterm/.portable_auto.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('已有建图任务，不能重复启动。',flush=True);return 3
    output=ROOT/'dev_ws/midterm/runs'/args.run_id
    if output.exists():print('编号已存在，请使用新编号。',flush=True);return 2
    problem=disk_problem()
    if problem:print('启动失败：'+problem,flush=True);return 2
    output.mkdir(parents=True,exist_ok=False)
    pcd=output/(args.run_id+'_native.pcd');status=output/(args.run_id+'_motion.json');log=output/'autonomous.log'
    cancelled=[False];child=None;node=None;log_handle=None;reader=None;result={};started=time.monotonic();code=1
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:cancelled.__setitem__(0,True))
    def publish(state,reason,**extra):
        payload={'state':state,'reason':reason,'run_id':args.run_id,'supervisor_pid':os.getpid(),'supervisor_start_ticks':process_start_ticks(os.getpid()),'heartbeat_monotonic':time.monotonic(),'output_directory':str(output),'elapsed_seconds':round(time.monotonic()-started,3),**extra}
        durable=state in ('FAILED','CANCELLED','COMPLETED','NEEDS_REVIEW')
        with result_lock(output):
            atomic_json(RUN_STATUS,payload,durable=durable);atomic_json(output/'result.json',payload,durable=durable)
        return payload
    def check_cancel():
        if cancelled[0] or STOP.exists():
            cancelled[0]=True;raise RuntimeError('operator_cancelled')
    def validate_command(argv,timeout,stdout=None):
        check_cancel()
        task=subprocess.Popen(argv,stdout=stdout,stderr=subprocess.STDOUT if stdout else None,start_new_session=True)
        deadline=time.monotonic()+timeout
        try:
            while task.poll() is None:
                with result_lock(output):
                    current=read_json(RUN_STATUS)
                    current['heartbeat_monotonic']=time.monotonic();atomic_json(RUN_STATUS,current,durable=False)
                check_cancel()
                if time.monotonic()>deadline:raise RuntimeError('validation_timeout:'+str(argv[1]))
                time.sleep(.2)
            check_cancel();return task.returncode
        finally:
            if task.poll() is None:stop_group(task)
    try:
        publish('STARTING','checking_live_inputs')
        check_cancel()
        if validate_command([sys.executable,str(ROOT/'deployment/check.py')],20)!=0:raise RuntimeError('preflight_failed')
        check_cancel()
        capture(output,{'run_id':args.run_id,'duration_seconds':args.duration,'speed_mps':args.speed,'stationary_mapping':os.environ.get('ROBOTAC_STATIONARY_MAPPING','1')=='1'})
        check_cancel()
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO);node=rclpy.create_node('robotac_run_supervisor');abort=node.create_client(Trigger,'/robotac/finals/abort')
        rviz='true' if os.environ.get('MIDTERM_RVIZ','1')=='1' and os.environ.get('DISPLAY') else 'false'
        argv=['ros2','launch',str(ROOT/'dev_ws/midterm/autonomous.launch.py'),'output_pcd:='+str(pcd),'status_path:='+str(status),'enable_rviz:='+rviz,'speed:='+str(args.speed),'hard_runtime:='+str(args.duration)]
        log_handle=log.open('w');child=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        def forward():
            for line in child.stdout:
                log_handle.write(line);log_handle.flush();print(line,end='',flush=True)
        reader=threading.Thread(target=forward,daemon=True);reader.start()
        publish('RUNNING','mission_started',launch_pid=child.pid)
        abort_requested=False;abort_sent=False;cancel_deadline=None;infrastructure_deadline=None;mission={};failure_reason=''
        while time.monotonic()-started<args.duration+95:
            rclpy.spin_once(node,timeout_sec=.2);now=time.monotonic();mission=read_json(status)
            problem=disk_problem(ROOT,MIN_SAVE_FREE_BYTES)
            if problem and not failure_reason:failure_reason=problem
            if cancelled[0] or STOP.exists() or failure_reason:
                if not abort_requested:
                    publish('STOPPING',failure_reason or 'operator_cancelled',launch_pid=child.pid)
                    print('正在停止并保存部分地图，请等待。',flush=True)
                    abort_requested=True;cancel_deadline=now+30
                if not abort_sent and abort.service_is_ready():abort.call_async(Trigger.Request());abort_sent=True
                if cancel_deadline and now>cancel_deadline:raise RuntimeError('cancel_save_timeout')
            if runtime_problem():
                if infrastructure_deadline is None:infrastructure_deadline=now+35
                if now>infrastructure_deadline:raise RuntimeError('runtime_failure_partial_save_timeout')
            state=mission.get('state')
            if state in ('MAP_SAVED','FAILED','CANCELLED'):break
            if child.poll() is not None:raise RuntimeError('required_algorithm_component_exited:'+str(child.returncode))
            publish('STOPPING' if abort_requested else 'SAVING' if state in ('SAVING','RETRYING') else 'RUNNING',mission.get('reason','launching'),launch_pid=child.pid)
        else:raise RuntimeError('mission_deadline_and_save_timeout')
        stop_group(child);child=None
        if reader:reader.join(timeout=3)
        terminal=mission.get('state')
        if failure_reason:
            terminal='FAILED';mission.update(state='FAILED',reason=failure_reason,partial_pcd_saved=pcd.exists())
            with result_lock(output):atomic_json(status,mission)
        elif abort_requested and terminal=='MAP_SAVED':
            terminal='CANCELLED';mission.update(state='CANCELLED',reason='operator_cancelled',partial_pcd_saved=pcd.exists())
            with result_lock(output):atomic_json(status,mission)
        if terminal!='MAP_SAVED':
            code=130 if terminal=='CANCELLED' else 4
            result=publish(terminal or 'FAILED',mission.get('reason','mission_failed'),partial_pcd=str(pcd) if pcd.exists() else None,mission_status=mission)
        else:
            # No successful run may hide a backend loss, even if an old map can be saved.
            problem=runtime_problem()
            if problem:raise RuntimeError(problem)
            check_cancel();verify(output)
            quality=output/(args.run_id+'_quality.json');preview=output/(args.run_id+'_preview.png')
            publish('VALIDATING','checking_saved_map',pcd=str(pcd),preview=str(preview),points=mission.get('saved_points'))
            with (output/'quality.log').open('w') as qlog:
                cmd=[sys.executable,str(ROOT/'dev_ws/tools/pcd_quality_report.py'),str(pcd),'--output',str(quality)]
                if validate_command(cmd,40,qlog)!=0:raise RuntimeError('native_pcd_quality_report_failed')
                if validate_command([sys.executable,str(ROOT/'finals_ws/test/validate_task1_artifact.py'),str(quality)],15,qlog)!=0:raise RuntimeError('native_pcd_basic_checks_failed')
            if validate_command([sys.executable,str(ROOT/'dev_ws/tools/pcd_preview.py'),str(pcd),'--output',str(preview),'--title',args.run_id+' native PCD'],40)!=0:raise RuntimeError('native_pcd_preview_failed')
            digest=hashlib.sha256(pcd.read_bytes()).hexdigest();(output/'SHA256SUMS.txt').write_text(digest+'  '+pcd.name+'\n')
            publish('VALIDATING','checking_geometry_coverage_and_outliers',pcd=str(pcd),preview=str(preview),sha256=digest,points=mission.get('saved_points'))
            code=validate_command([sys.executable,str(ROOT/'deployment/audit_run.py'),args.run_id,'--owner-pid',str(os.getpid()),'--owner-start-ticks',str(process_start_ticks(os.getpid()))],65)
            if code not in (0,5):raise RuntimeError('geometry_audit_execution_failed')
            with result_lock(output):
                result=read_json(output/'result.json');atomic_json(RUN_STATUS,result)
            print('原生点云已保存：'+str(pcd),flush=True)
    except Exception as e:
        reason=str(e);print('建图失败：'+reason,file=sys.stderr,flush=True)
        prior=read_json(status);prior.update(state='CANCELLED' if cancelled[0] else 'FAILED',reason=reason)
        with result_lock(output):atomic_json(status,prior)
        checkpoints=sorted({str(x) for pattern in ('*.checkpoint.pcd','*.fault_checkpoint.pcd') for x in output.glob(pattern)})
        result=publish(prior['state'],reason,partial_pcd=str(pcd) if pcd.exists() else None,checkpoint_files=checkpoints);code=130 if cancelled[0] else 2 if reason=='preflight_failed' else 4
    finally:
        if child:stop_group(child)
        if reader:reader.join(timeout=3)
        if log_handle:log_handle.close()
        if node:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    print('本轮状态：'+str(result.get('state'))+'；结果记录：'+str(output/'result.json'),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
