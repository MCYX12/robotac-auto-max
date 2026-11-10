#!/usr/bin/env python3
"""Bounded readiness checks before any control is started."""
import argparse,json,sys,time
from pathlib import Path
import rclpy
from ament_index_python.packages import get_package_prefix
from common import ROOT,atomic_json,runtime_problem,disk_problem
from telemetry import Telemetry

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--timeout',type=float,default=12.);args=parser.parse_args()
    if not 1<=args.timeout<=60:raise SystemExit('timeout must be 1..60 seconds')
    problem=disk_problem()
    if problem:print('检查失败：'+problem,flush=True);return 2
    for package in ('robotac_finals','robotac_mapping_baseline'):get_package_prefix(package)
    sys.path.insert(0,str(ROOT/'deps/zsibot_sdk/lib/zsl-1/x86_64'))
    import mc_sdk_zsl_1_py
    problem=runtime_problem(require_running=True)
    if problem:print('检查失败：'+problem+'。请先启动仿真。',flush=True);return 2
    rclpy.init();node=Telemetry('robotac_preflight');end=time.monotonic()+args.timeout
    while time.monotonic()<end:
        rclpy.spin_once(node,timeout_sec=.2)
        if node.ready():break
    problem=runtime_problem(require_running=True);ok=node.ready() and not problem
    result={'sensor_check_pass':ok,'reason':problem or ('' if ok else 'missing_or_invalid_live_sensors'),**node.summary()}
    atomic_json(ROOT/'deployment/preflight.json',result)
    if not ok:print('检查失败：'+result['reason']+'；缺少新鲜的有效数据：'+','.join(node.stale()),flush=True)
    node.destroy_node();rclpy.shutdown()
    if ok:print('检查通过，可以开始建图。',flush=True)
    return 0 if ok else 2
if __name__=='__main__':raise SystemExit(main())
