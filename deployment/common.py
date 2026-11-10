"""Shared bounded process lifecycle and atomic status helpers."""
import json,os,signal,time,math,fcntl,shutil
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STOP=ROOT/'deployment/stop.request'
MIN_START_FREE_BYTES=2*1024**3
MIN_SAVE_FREE_BYTES=512*1024**2

@contextmanager
def result_lock(folder):
    """Serialize only short result commits, never a long audit or ROS wait."""
    with (Path(folder)/'.result.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)

def disk_problem(path=ROOT,minimum=MIN_START_FREE_BYTES):
    available=shutil.disk_usage(path).free
    return f'insufficient_disk_space:free={available},required={minimum}' if available<minimum else ''

def runtime_duration(value):
    try:value=float(value)
    except (TypeError,ValueError):raise ValueError('运行时间必须是 30..1800 秒的有限数值')
    if not math.isfinite(value) or not 30<=value<=1800:raise ValueError('运行时间必须是 30..1800 秒的有限数值')
    return value

def process_start_ticks(pid):
    try:return int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
    except (OSError,ValueError,IndexError):return None

def supervisor_alive(status,script):
    pid=status.get('supervisor_pid')
    if not process_alive(pid):return False
    expected=status.get('supervisor_start_ticks')
    if expected is not None and expected!=process_start_ticks(pid):return False
    try:return script.encode() in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
    except OSError:return False

def heartbeat_fresh(status,limit=5.):
    try:stamp=float(status.get('heartbeat_monotonic',0));now=time.monotonic()
    except (TypeError,ValueError):return False
    return math.isfinite(stamp) and now-limit<=stamp<=now+1.
def atomic_json(path,payload,durable=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(payload,f,ensure_ascii=False,indent=2);f.write('\n');f.flush()
        if durable:os.fsync(f.fileno())
    os.replace(tmp,path)
def read_json(path):
    try:
        value=json.loads(Path(path).read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError,TypeError):return {}
def process_alive(pid):
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<=0:return False
    try:
        os.kill(pid,0)
        return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (OSError,ValueError,TypeError,IndexError):return False
def stop_group(child,first=signal.SIGINT):
    # Always signal the recorded process group, even when its wrapper already exited.
    for sig,timeout in [(first,6),(signal.SIGTERM,3),(signal.SIGKILL,1)]:
        try:
            if sig==signal.SIGINT and child.poll() is None:child.send_signal(sig)
            else:os.killpg(child.pid,sig)
        except ProcessLookupError:return
        try:child.wait(timeout=timeout)
        except __import__('subprocess').TimeoutExpired:pass
        try:os.killpg(child.pid,0)
        except ProcessLookupError:return
    try:child.wait(timeout=1)
    except __import__('subprocess').TimeoutExpired:pass
def runtime_problem(status=None,require_running=False):
    s=read_json(ROOT/'deployment/runtime_status.json') if status is None else status
    if not isinstance(s,dict):return 'invalid_runtime_status'
    if s.get('state') not in (('RUNNING',) if require_running else ('RUNNING','STOPPING')):return 'runtime_not_ready:'+str(s.get('reason',s.get('state','missing')))
    try:heartbeat=float(s.get('heartbeat_monotonic',0))
    except (TypeError,ValueError):return 'invalid_runtime_heartbeat'
    if not math.isfinite(heartbeat):return 'invalid_runtime_heartbeat'
    if time.monotonic()-heartbeat>4 or heartbeat>time.monotonic()+1:return 'runtime_heartbeat_stale'
    if not supervisor_alive(s,str(ROOT/'deployment/simulation.py')):return 'runtime_supervisor_gone'
    return ''
