"""Additional review of a flagged fixed patch, preserving its original verdict.

A stationary main component can coexist with a changing proportion of facade
returns. This cannot certify zero drift or dismiss changes of other components.
No scene reference, registration, modified points, or configurable thresholds.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation
from temporal_consistency import load_captures


POLICY = dict(core_band_m=.015, minimum_core_points=30, minimum_core_frames=3,
              minimum_core_points_per_frame=5, maximum_core_width_m=.01,
              maximum_core_change_m=.005, minimum_initial_core_fraction=.50,
              minimum_late_core_fraction=.25, minimum_fraction_change=.10,
              maximum_raw_filtered_change_difference_m=.005)
EXPLAINED = 'STATIONARY_MAIN_COMPONENT_WITH_VIEW_COMPOSITION'


def visit_summary(frames):
    frames = [np.asarray(v, dtype=float) for v in frames]
    frames = [v[np.isfinite(v)] for v in frames if len(v)]
    if not frames:
        return None
    core = [v[np.abs(v) <= POLICY['core_band_m']] for v in frames]
    points = np.concatenate(frames)
    core_points = np.concatenate(core)
    supported = [v for v in core if len(v) >= POLICY['minimum_core_points_per_frame']]
    broad = [float(np.median(v)) for v in frames if len(v) >= 15]
    return dict(points=len(points), core_points=len(core_points), core_frames=len(supported),
                core_fraction=len(core_points)/len(points),
                broad_median=float(np.median(broad)) if broad else None,
                core_median=float(np.median([np.median(v) for v in supported])) if supported else None,
                core_width=float(np.ptp(np.percentile(core_points, [5, 95]))) if len(core_points) else None)


def classify(raw_visits, filtered_visits):
    result = dict(status='INSUFFICIENT_PAIRED_COMPONENT_SUPPORT', policy=POLICY,
                  raw=[visit_summary(v) for v in raw_visits],
                  filtered=[visit_summary(v) for v in filtered_visits])
    if len(raw_visits) != len(filtered_visits) or len(raw_visits) < 2:
        return result
    for source in ('raw', 'filtered'):
        for i, v in enumerate(result[source]):
            fraction = POLICY['minimum_initial_core_fraction'] if i == 0 else POLICY['minimum_late_core_fraction']
            if (v is None or v['core_points'] < POLICY['minimum_core_points'] or
                    v['core_frames'] < POLICY['minimum_core_frames'] or
                    v['core_fraction'] < fraction or v['broad_median'] is None):
                return result
            if v['core_width'] > POLICY['maximum_core_width_m']:
                result['status'] = 'MAIN_COMPONENT_TOO_BROAD'
                return result
    raw, filtered = result['raw'], result['filtered']
    changes = []
    for i in range(1, len(filtered)):
        raw_change = raw[i]['broad_median']-raw[0]['broad_median']
        filtered_change = filtered[i]['broad_median']-filtered[0]['broad_median']
        core_change = max(abs(v[i]['core_median']-v[0]['core_median']) for v in (raw, filtered))
        fraction_change = min(abs(v[i]['core_fraction']-v[0]['core_fraction']) for v in (raw, filtered))
        changes.append(dict(raw_broad_change_m=raw_change, filtered_broad_change_m=filtered_change,
                            maximum_core_change_m=core_change, minimum_core_fraction_change=fraction_change))
        if core_change > POLICY['maximum_core_change_m']:
            result['status'] = 'MAIN_COMPONENT_CHANGE_REQUIRES_REVIEW'
            break
        if abs(raw_change-filtered_change) > POLICY['maximum_raw_filtered_change_difference_m']:
            result['status'] = 'RAW_FILTERED_DISAGREEMENT'
            break
        if abs(filtered_change) > .01 and fraction_change < POLICY['minimum_fraction_change']:
            result['status'] = 'NO_CORROBORATED_COMPOSITION_CHANGE'
            break
    else:
        result['status'] = EXPLAINED
    result['changes'] = changes
    return result


def residual_visits(data, patch, permitted_stamps):
    normal = np.asarray(patch['normal_xy'])
    tangent = np.array([-normal[1], normal[0]])
    ub, zb = patch['tangent_bounds_m'], patch['height_bounds_m']
    lookup = {int(s): i for i, s in enumerate(data['frame_stamps_ns'])}
    visits = []
    for stamps in permitted_stamps:
        frames = []
        for stamp in stamps:
            if stamp not in lookup:
                continue
            i = lookup[stamp]
            a, b = data['frame_offsets'][i:i+2]
            pose = data['frame_interpolated_odom'][i]
            xyz = Rotation.from_quat(pose[3:7]).apply(data['local_xyz'][a:b]+[0, 0, .3])+pose[:3]
            u = xyz[:, :2]@tangent
            r = xyz[:, :2]@normal+patch['plane_d_m']
            within = ((u >= ub[0]) & (u <= ub[1]) & (xyz[:, 2] >= zb[0]) &
                      (xyz[:, 2] <= zb[1]) & (np.abs(r) <= .20))
            frames.append(r[within])
        visits.append(frames)
    return visits


def review(folder, report_path, output):
    report = json.loads(report_path.read_text())
    raw, raw_integrity = load_captures(sorted((folder/'temporal').glob('chunk_*.npz')), True)
    filtered, filtered_integrity = load_captures(sorted((folder/'temporal_filtered').glob('chunk_*.npz')), True)
    identities = [x['source'] for x in (raw_integrity, filtered_integrity)]
    assert all(x['run_id'] == folder.name for x in identities)
    assert identities[0]['runtime_generation'] == identities[1]['runtime_generation']
    assert identities[0]['cloud_topic'] == '/front_lidar'
    assert identities[1]['cloud_topic'] == '/robotac/filtered_lidar'
    assert report['capture_integrity'] == filtered_integrity
    rows = []
    for patch in report['patches']:
        if not (patch['normal_change_review'] or patch['thickness_change_review']):
            continue
        visits = [patch['matched_visits'][0]]+[v for v in patch['matched_visits'][1:]
                  if v['start_s']-patch['matched_visits'][0]['end_s'] >= 30.]
        stamps = [[int(m['stamp_ns']) for m in patch['measurements']
                   if v['start_s'] <= m['time_s'] <= v['end_s']]
                  for v in visits]
        if patch['thickness_change_review']:
            result = dict(status='UNRESOLVED_THICKNESS_CHANGE')
        else:
            result = classify(residual_visits(raw, patch, stamps), residual_visits(filtered, patch, stamps))
        rows.append(dict(cell=patch['cell'], original_normal_change_review=patch['normal_change_review'],
                         original_thickness_change_review=patch['thickness_change_review'], review=result))
    result = dict(run_id=folder.name, runtime_generation=identities[0]['runtime_generation'], policy=POLICY,
                  filtered_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                  raw_capture_integrity=raw_integrity, filtered_capture_integrity=filtered_integrity,
                  original_flags_preserved=True, reviews=rows,
                  all_flagged_patches_corroborated=all(r['review']['status'] == EXPLAINED for r in rows),
                  limitation='Only the frozen main component and source agreement are reviewed. Original broad-patch flags remain; other components, unobserved support and global zero drift are not certified.')
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-folder', required=True, type=Path)
    parser.add_argument('--filtered-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = review(args.run_folder, args.filtered_report, args.output)
    print(json.dumps(dict(run_id=result['run_id'], all_flagged_patches_corroborated=result['all_flagged_patches_corroborated'],
                          reviews=[dict(cell=r['cell'], status=r['review']['status']) for r in result['reviews']])))
