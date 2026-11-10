#!/usr/bin/env python3
"""Offline, fail-closed geometry proxy audit for one already saved native map."""
import argparse, fcntl, hashlib, re, subprocess, sys, time
from pathlib import Path
from common import ROOT, STOP, atomic_json, read_json, result_lock

ALLOWED_STATES = ('COMPLETED', 'VALIDATING', 'NEEDS_REVIEW')
REQUIRED_CHECKS = {
    'floor_coverage_at_10cm_ge_99_5pct',
    'floor_coverage_at_20cm_ge_99_999pct',
    'largest_floor_gap_radius_le_25cm',
    'wall_coverage_at_15cm_ge_99_5pct',
    'wall_median_reference_residual_le_5mm',
    'wall_outliers_beyond_20cm_le_0_01pct',
    'all_height_footprint_outliers_beyond_35cm_le_0_01pct',
    'all_points_within_configured_height_bounds',
    'each_wall_coverage_at_20cm_ge_99_5pct',
    'each_wall_largest_gap_le_25cm',
    'top_strip_coverage_at_20cm_ge_99_5pct',
    'top_strip_largest_gap_le_25cm',
}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(result):
    return tuple(result.get(key) for key in ('run_id', 'supervisor_pid', 'supervisor_start_ticks', 'sha256'))

def validate_report(report, digest, sources):
    checks = report.get('proxy_checks')
    if not isinstance(checks, dict) or set(checks) != REQUIRED_CHECKS or any(type(v) is not bool for v in checks.values()):
        raise ValueError('geometry audit returned invalid checks')
    passed = all(checks.values())
    if type(report.get('proxy_check_pass')) is not bool or report['proxy_check_pass'] != passed:
        raise ValueError('geometry verdict does not match its checks')
    if report.get('quality_state') != ('PASS_PROXY_CHECKS' if passed else 'NEEDS_REVIEW'):
        raise ValueError('geometry audit returned inconsistent quality state')
    expected = {'native_sha256': digest, 'reference_sha256': sources['reference'],
                'auditor_sha256': sources['auditor'], 'pcd_reader_sha256': sources['reader']}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('geometry report provenance does not match audited inputs')
    if not isinstance(report.get('limitations'), list) or not report.get('audit_policy'):
        raise ValueError('geometry audit omitted its policy or limitations')
    return passed

def audit_run(folder, owner_pid=None, owner_start_ticks=None):
    folder = folder.resolve(); base = (ROOT/'dev_ws/midterm/runs').resolve()
    if folder.parent != base or not folder.is_dir():
        raise ValueError('result directory must be an existing direct child of project runs')
    with (folder/'.audit.lock').open('a') as audit_lock:
        try:
            fcntl.flock(audit_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('an audit for this run is already in progress')
        return _audit_locked(folder, owner_pid, owner_start_ticks)

def _audit_locked(folder, owner_pid, owner_start_ticks):
    pcd = folder/(folder.name+'_native.pcd')
    with result_lock(folder):
        initial = read_json(folder/'result.json')
        if initial.get('state') not in ALLOWED_STATES or initial.get('run_id') != folder.name:
            raise ValueError('only a healthy saved acquisition can enter the geometry audit')
        if initial['state'] == 'VALIDATING' and (
            owner_pid is None or owner_start_ticks is None or
            initial.get('supervisor_pid') != owner_pid or initial.get('supervisor_start_ticks') != owner_start_ticks
        ):
            raise ValueError('active validation belongs to the run supervisor; wait for its result')
        initial_identity = identity(initial)
    digest = sha256(pcd)
    if not initial.get('sha256') or digest != initial['sha256']:
        raise ValueError('native PCD changed after acquisition or acquisition checksum is missing')
    paths = {'reference': ROOT/'runtime/src/robot_mujoco/zsibot_robots/xgb/unreal.xml',
             'auditor': ROOT/'deployment/validation/map_audit.py',
             'reader': ROOT/'dev_ws/tools/pcd_quality_report.py'}
    sources = {key: sha256(path) for key, path in paths.items()}
    geometry = folder/'geometry_audit.json'
    with (folder/'geometry_audit.log').open('w') as log:
        subprocess.run([sys.executable, str(paths['auditor']), str(pcd), '--output', str(geometry)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    report = read_json(geometry)
    passed = validate_report(report, digest, sources)
    if sha256(pcd) != digest or any(sha256(paths[key]) != value for key, value in sources.items()):
        raise ValueError('PCD, reference or audit implementation changed during validation')
    state = 'COMPLETED' if passed else 'NEEDS_REVIEW'
    with result_lock(folder):
        result = read_json(folder/'result.json')
        motion = folder/(folder.name+'_motion.json'); m = read_json(motion)
        if result.get('state') not in ALLOWED_STATES or m.get('state') in ('FAILED', 'CANCELLED'):
            raise ValueError('acquisition failed or was cancelled while auditing; verdict not committed')
        if identity(result) != initial_identity:
            raise ValueError('run ownership or native checksum changed while auditing')
        active = read_json(ROOT/'deployment/run_status.json')
        same_active = active.get('run_id') == folder.name
        if same_active and (active.get('state') in ('FAILED', 'CANCELLED', 'STOPPING') or
                            (result.get('state') == 'VALIDATING' and STOP.exists())):
            raise ValueError('active run stopped or was cancelled while auditing; verdict not committed')
        if same_active and identity(active) != initial_identity:
            raise ValueError('active supervisor ownership changed while auditing')
        if not (folder/'pre_geometry_result.json').exists():
            atomic_json(folder/'pre_geometry_result.json', result)
        result.update(state=state,
            reason='configured_geometry_proxy_checks_passed' if passed else 'geometry_proxy_checks_failed',
            quality_state=report['quality_state'], geometry_audit=str(geometry), sha256=digest, pcd=str(pcd),
            audit_policy=report['audit_policy'], audit_sources_sha256=sources,
            perfect_map_certified=False, quality_limits=report['limitations'], heartbeat_monotonic=time.monotonic())
        atomic_json(folder/'result.json', result)
        if same_active: atomic_json(ROOT/'deployment/run_status.json', result)
        m.update(state=state, reason=result['reason'], quality_validation=report['quality_state'], perfect_map_certified=False)
        atomic_json(motion, m)
    rows = ['# 本轮地图质量检查', '', f'编号：{folder.name}',
        f'结果：{"通过设定的几何代理检查" if passed else "未通过，需继续修复或补扫"}。这不等于完美地图认证。', '',
        '| 检查 | 实测 |', '|---|---:|',
        f'| 地面覆盖（距实测点≤10厘米） | {report["floor_coverage"]["0.1"]*100:.4f}% |',
        f'| 地面覆盖（≤20厘米） | {report["floor_coverage"]["0.2"]*100:.4f}% |',
        f'| 墙面覆盖（≤15厘米） | {report["interior_wall_surface_coverage"]["0.15"]*100:.4f}% |',
        f'| 最差单墙覆盖（≤20厘米） | {min(w["surface_coverage_20cm"] for w in report["worst_walls"])*100:.4f}% |',
        f'| 最大地面缺口半径 | {report["floor_largest_gap_radius_m"]:.3f} 米 |',
        f'| 最大单墙缺口距离 | {max(w["max_missing_distance_m"] for w in report["worst_walls"]):.3f} 米 |',
        f'| 3米碰撞墙顶部条带覆盖（≤20厘米） | {report["additional_surface_diagnostics"]["wall_upper_strip"]["coverage_at_20cm"]*100:.4f}% |',
        f'| 顶部条带最大缺口距离 | {report["additional_surface_diagnostics"]["wall_upper_strip"]["maximum_gap_m"]:.3f} 米 |',
        f'| 距简化墙体超过20厘米的点 | {report["wall_point_fraction_beyond"]["0.2"]*100:.5f}% |',
        f'| 全高度超出场景足迹35厘米的点 | {report["all_point_spatial_support"]["outside_footprint_points"]}点 / {report["all_point_spatial_support"]["outside_footprint_fraction"]*100:.5f}% |', '',
        '参照物是本机场景的简化碰撞模型，只用于离线核对。它与可见墙面凸边存在差异；覆盖率采用指定距离容差。没有用参考模型补点，也没有对原生点云做配准掩盖误差。', '',
        '墙体顶部条带按3米高碰撞墙检查观察是否充分；端面、底部条带和空间异常簇单独报告。通过这些代理检查不等于完美地图或零漂移证明。', '',
        '不通过项：'+('无' if passed else '、'.join(k for k,v in report['proxy_checks'].items() if not v)), '',
        f'原生点云 SHA256：`{digest}`。', '详细缺口坐标、逐墙检查和校验来源见 geometry_audit.json，图示见 geometry_audit.png。']
    (folder/'地图质量检查.md').write_text('\n'.join(rows)+'\n')
    print(('地图质量代理检查通过' if passed else '地图质量未通过：不能作为完整合格地图')+'；报告：'+str(folder/'地图质量检查.md'), flush=True)
    return 0 if passed else 5

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('run_id')
    parser.add_argument('--owner-pid', type=int); parser.add_argument('--owner-start-ticks', type=int)
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}',args.run_id):parser.error('invalid run ID')
    try:return audit_run(ROOT/'dev_ws/midterm/runs'/args.run_id, args.owner_pid, args.owner_start_ticks)
    except Exception as error:
        print('质量检查失败：'+str(error),file=sys.stderr,flush=True);return 2
if __name__=='__main__':raise SystemExit(main())
