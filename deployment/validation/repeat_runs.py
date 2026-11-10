#!/usr/bin/env python3
"""Host-side repeatability check using the normal Docker start/stop entrypoints.

Run explicitly: python3 deployment/validation/repeat_runs.py --count 2
Every attempt is retained. There are no hidden retries or discarded failures.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'deployment'))
sys.path.insert(0, str(ROOT / 'dev_ws/tools'))
from common import atomic_json, read_json


def compare(first, second):
    """Native coordinates only; symmetric nearest measured wall point distances."""
    import numpy as np
    from scipy.spatial import cKDTree
    from pcd_quality_report import load_binary_pcd

    clouds = []
    for folder in (first, second):
        points, _ = load_binary_pcd(folder / (folder.name + '_native.pcd'))
        wall = points[(points[:, 2] > .25) & (points[:, 2] < 2.85)]
        if len(wall) < 1000:
            raise ValueError('insufficient_measured_wall_points_for_comparison')
        clouds.append(wall)
    directions = []
    for source, target in (clouds, clouds[::-1]):
        sample = source[::max(1, len(source) // 200000)]
        distance = cKDTree(target).query(sample, workers=2)[0]
        directions.append({'sample_points': len(sample),
                           'median_m': float(np.median(distance)),
                           'p95_m': float(np.percentile(distance, 95)),
                           'p99_m': float(np.percentile(distance, 99)),
                           'maximum_m': float(distance.max())})
    # These limits are fixed before runs. They describe repeatability, not
    # absolute drift or exact surface completeness. Individual geometry gates
    # must also pass independently on every run.
    passed = all(d['p95_m'] <= .10 and d['p99_m'] <= .20 for d in directions)
    return {'first_run': first.name, 'second_run': second.name,
            'alignment': 'none', 'wall_z_band_m': [.25, 2.85],
            'limits': {'p95_m': .10, 'p99_m': .20},
            'directions': directions, 'pass': passed,
            'limitation': 'Across-run consistency in native coordinates is not a measurement of within-run temporal drift.'}


def check_result(folder, code):
    result = read_json(folder / 'result.json')
    proof = read_json(folder / 'provenance.json')
    proof_check = read_json(folder / 'provenance_check.json')
    geometry = read_json(folder / 'geometry_audit.json')
    pcd = folder / (folder.name + '_native.pcd')
    digest = hashlib.sha256(pcd.read_bytes()).hexdigest() if pcd.is_file() else None
    checks = {
        'exit_success': code == 0,
        'terminal_completed': result.get('state') == 'COMPLETED',
        'geometry_pass': geometry.get('proxy_check_pass') is True,
        'geometry_checks_nonempty': bool(geometry.get('proxy_checks')),
        'every_geometry_check_pass': bool(geometry.get('proxy_checks')) and all(v is True for v in geometry['proxy_checks'].values()),
        'native_digest_matches': digest is not None and digest == result.get('sha256'),
        'geometry_native_digest_matches': digest is not None and digest == geometry.get('native_sha256'),
        'geometry_source_digests_match': bool(result.get('audit_sources_sha256')) and all(
            geometry.get(key) == result['audit_sources_sha256'].get(source)
            and isinstance(geometry.get(key), str) and len(geometry[key]) == 64
            for key, source in [('reference_sha256', 'reference'), ('auditor_sha256', 'auditor'), ('pcd_reader_sha256', 'reader')]),
        'geometry_policy_matches': bool(result.get('audit_policy')) and result['audit_policy'] == geometry.get('audit_policy'),
        'version_unchanged': proof_check.get('pass') is True,
        'version_record_present': bool(proof.get('software_sha256')),
    }
    return {'run_id': folder.name, 'exit_code': code, 'state': result.get('state'),
            'reason': result.get('reason'), 'sha256': digest,
            'points': result.get('points'), 'checks': checks,
            'pass': all(checks.values()), 'geometry': geometry,
            'software_sha256': proof.get('software_sha256', {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--duration', type=int, default=1800)
    parser.add_argument('--prefix', default='repeat_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    import re
    if not 2 <= args.count <= 5 or not 30 <= args.duration <= 1800:
        parser.error('count must be 2..5 and duration 30..1800')
    if not re.fullmatch('[A-Za-z0-9_-]{1,60}', args.prefix):
        parser.error('prefix must use 1..60 ASCII letters, digits, hyphens or underscores')
    output = ROOT / 'deployment/validation' / args.prefix
    output.mkdir(exist_ok=False)
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'state': 'RUNNING', 'requested_runs': args.count,
               'duration_seconds_each': args.duration,
               'scope': 'local_simulation', 'perfect_map_certified': False,
               'runs': [], 'comparisons': [], 'acceptance': 'Each run must pass all geometry and provenance checks; all native-coordinate wall comparisons must have p95<=0.10m and p99<=0.20m in both directions.'}
    cancelled = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    def write():
        summary['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / 'summary.json', summary)
    child = None
    try:
        write()
        for number in range(1, args.count + 1):
            if cancelled[0]:
                raise KeyboardInterrupt
            runid = f'{args.prefix}_{number:02d}'
            summary['active_run'] = runid
            write()
            env = os.environ.copy()
            env.update(ROBOTAC_HEADLESS='1', MIDTERM_RVIZ='0', ROBOTAC_RUNTIME_SECONDS=str(args.duration))
            print(f'Starting {number}/{args.count}: {runid}', flush=True)
            with (output / (runid + '.log')).open('w') as log:
                child = subprocess.Popen(['bash', str(ROOT / 'docker.sh'), 'start', runid], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + args.duration + 240
                while child.poll() is None:
                    if cancelled[0] or time.monotonic() > deadline:
                        subprocess.run(['bash', str(ROOT / 'docker.sh'), 'stop'], timeout=75, check=False)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGTERM)
                            child.wait(timeout=15)
                        if cancelled[0]:
                            raise KeyboardInterrupt
                        raise RuntimeError('repeat_run_exceeded_bounded_timeout')
                    time.sleep(1)
            folder = ROOT / 'dev_ws/midterm/runs' / runid
            item = check_result(folder, child.returncode)
            child = None
            summary['runs'].append(item)
            write()
            print(f'{runid}: {item["state"]}, checks={item["pass"]}', flush=True)
            if not item['pass']:
                raise RuntimeError('run_failed_acceptance:' + runid)
            if number > 1:
                for first in summary['runs'][:-1]:
                    if first['software_sha256'] != item['software_sha256']:
                        raise RuntimeError('software_changed_between_runs')
                    comparison = compare(ROOT / 'dev_ws/midterm/runs' / first['run_id'], folder)
                    summary['comparisons'].append(comparison)
                    write()
                    if not comparison['pass']:
                        raise RuntimeError('native_map_repeatability_check_failed')
        summary['state'] = 'PASS' if args.duration == 1800 else 'SHORT_RUN_CHECKS_PASS'
        summary.pop('active_run', None)
        write()
        return 0
    except KeyboardInterrupt:
        summary.update(state='CANCELLED', reason='operator_cancelled')
        write()
        return 130
    except Exception as error:
        summary.update(state='FAILED', reason=str(error))
        write()
        print(str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        if child is not None and child.poll() is None:
            subprocess.run(['bash', str(ROOT / 'docker.sh'), 'stop'], timeout=75, check=False)
        print('Repeatability report: ' + str(output / 'summary.json'), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
