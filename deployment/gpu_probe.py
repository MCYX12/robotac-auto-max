"""Poll NVML outside the ROS/heartbeat loop, with a bounded child lifetime."""
import subprocess


class GpuProbe:
    def __init__(self, interval=10., timeout=2., command=None):
        self.interval=interval
        self.timeout=timeout
        self.command=command or ['nvidia-smi','--query-gpu=name','--format=csv,noheader']
        self.process=None
        self.started=None
        self.last_success=None

    def tick(self, now):
        if self.process is not None:
            code=self.process.poll()
            if code is None:
                if now-self.started > self.timeout:
                    self.close()
                    raise RuntimeError('GPU_probe_timeout')
            else:
                out,err=self.process.communicate(timeout=.2)
                self.process=None
                if code or not out.strip() or 'Failed' in out:
                    raise RuntimeError('GPU_access_lost:'+err.strip()[:160]+out.strip()[:160])
                self.last_success=now
        if self.process is None and (self.started is None or now-self.started >= self.interval):
            self.process=subprocess.Popen(self.command,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            self.started=now
        return self.last_success is not None

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:self.process.kill()
            try:self.process.communicate(timeout=1.)
            except subprocess.TimeoutExpired:
                self.process.stdout.close();self.process.stderr.close()
            self.process=None
