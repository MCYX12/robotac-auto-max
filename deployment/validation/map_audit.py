#!/usr/bin/env python3
"""Offline reference-only audit. Never imported by mapping or navigation nodes."""
import argparse,hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'dev_ws/tools'))
from pcd_quality_report import load_binary_pcd

# Frozen before the next acquisition; the original six thresholds are unchanged.
# The 0.35 m footprint margin covers visible 0.05..0.15 m trim, a 0.05 m
# reference grid and modest collision/render mismatch. It is not a drift tolerance.
AUDIT_POLICY = {
    'version': '2026-11-03.v3',
    'footprint_margin_m': .35,
    'maximum_footprint_outlier_fraction': .0001,
    'minimum_per_wall_coverage_at_20cm': .995,
    'maximum_per_wall_gap_m': .25,
    'minimum_z_m': -.30,
    'maximum_z_m': 3.35,
    'outlier_cluster_voxel_m': .25,
    'top_strip_heights_m': [2.75, 2.85, 2.95],
    'minimum_top_strip_coverage_at_20cm': .995,
    'maximum_top_strip_gap_m': .25,
}

def spatial_outlier_clusters(points, voxel_size=.25):
    """Report coherent anomalies separately from a density-weighted global rate."""
    if not len(points):return []
    if np.any(np.abs(points/voxel_size)>=np.iinfo(np.int64).max):
        raise ValueError('spatial outlier coordinates exceed supported clustering range')
    cells,counts=np.unique(np.floor(points/voxel_size).astype(np.int64),axis=0,return_counts=True)
    remaining={tuple(c):int(n) for c,n in zip(cells,counts)};clusters=[]
    neighbors=[(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if (x,y,z)!=(0,0,0)]
    while remaining:
        first=next(iter(remaining));todo=[first];members=[];total=0
        while todo:
            cell=todo.pop();count=remaining.pop(cell,None)
            if count is None:continue
            total+=count;members.append(cell)
            for delta in neighbors:
                nxt=tuple(a+b for a,b in zip(cell,delta))
                if nxt in remaining:todo.append(nxt)
        occupied=np.asarray(members)
        clusters.append({'points':total,'occupied_25cm_voxels':len(members),
                         'bbox_min_m':(occupied.min(axis=0)*voxel_size).tolist(),
                         'bbox_max_m':((occupied.max(axis=0)+1)*voxel_size).tolist()})
    return sorted(clusters,key=lambda c:c['points'],reverse=True)

def boxes_from_xml(path):
    boxes=[]
    world=ET.parse(path).getroot().find('worldbody')
    if world is None:raise ValueError('reference has no worldbody')
    for g in world.findall('geom'):
        if g.get('type')!='box':continue
        pos=np.fromstring(g.get('pos'),sep=' ');size=np.fromstring(g.get('size'),sep=' ')
        quat=np.fromstring(g.get('quat','1 0 0 0'),sep=' ')
        if len(pos)!=3 or len(size)!=3 or len(quat)!=4 or not np.isfinite(np.r_[pos,size,quat]).all() or np.any(size<=0):
            raise ValueError('invalid reference box')
        if abs(np.linalg.norm(quat)-1)>.001 or abs(quat[1])+abs(quat[2])>.001:
            raise ValueError('only normalized yaw-only reference boxes are supported')
        yaw=2*np.arctan2(quat[3],quat[0]);c,s=np.cos(yaw),np.sin(yaw)
        boxes.append((g.get('name'),pos,size,np.array([[c,-s],[s,c]])))
    if not boxes:raise ValueError('reference contains no supported wall boxes')
    if len({box[0] for box in boxes})!=len(boxes):raise ValueError('reference wall names must be unique')
    return boxes

def scene_grid(boxes,res=.05):
    corners=np.concatenate([pos[:2]+np.array([[a,b] for a in (-size[0],size[0]) for b in (-size[1],size[1])])@rot.T for _,pos,size,rot in boxes])
    low=np.floor((corners.min(axis=0)-2)/res)*res;high=np.ceil((corners.max(axis=0)+2)/res)*res
    xs=np.arange(low[0],high[0],res)+res/2;ys=np.arange(low[1],high[1],res)+res/2
    xx,yy=np.meshgrid(xs,ys);xy=np.column_stack([xx.ravel(),yy.ravel()]);occupied=np.zeros(len(xy),bool)
    for _,pos,size,rot in boxes:
        local=(xy-pos[:2])@rot
        occupied|=(np.abs(local)<=size[:2]+1e-6).all(axis=1)
    occupied=occupied.reshape(xx.shape);free=~occupied
    labels,_=ndimage.label(free);ij=np.floor((np.array([0,0])-low)/res).astype(int)
    start_label=labels[ij[1],ij[0]]
    if start_label==0:raise ValueError('reference start lies inside a collision box')
    component=labels==start_label
    touches=component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any()
    if touches:
        # Small gaps between exported collision boxes are not traversable by the robot.
        # Determine the component of robot-center free space, then expand toward walls
        # only through free cells for one robot radius; report this approximation.
        clearance=ndimage.distance_transform_edt(free)*res
        core=clearance>.32;labels,_=ndimage.label(core)
        label=labels[ij[1],ij[0]]
        if label==0:raise RuntimeError('Reference start is not free')
        component=labels==label
        if component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():raise RuntimeError('Robot-accessible reference is open; coverage cannot be certified')
        for _ in range(int(.32/res)):component=ndimage.binary_dilation(component,mask=free)
        print('REFERENCE_NOTE: collision seams sealed using robot-center clearance .32m; floor mask is a geometric proxy',flush=True)
    return low,res,xx,yy,occupied,component

def audit(path,xml,out):
    path=Path(path);xml=Path(xml);out=Path(out)
    native_digest=hashlib.sha256(path.read_bytes()).hexdigest()
    reference_digest=hashlib.sha256(xml.read_bytes()).hexdigest()
    points,metadata=load_binary_pcd(path)
    if len(points)<1000 or len(points)!=int(metadata['POINTS'][0]) or not np.isfinite(points).all():
        raise ValueError('native PCD must have matching counts and only finite XYZ points')
    boxes=boxes_from_xml(xml);low,res,xx,yy,occupied,floor_mask=scene_grid(boxes)
    if any(abs(pos[2]-size[2])>.001 or abs(pos[2]+size[2]-3.)>.001 for _,pos,size,_ in boxes):
        raise ValueError('this audit policy requires the configured 0..3 m reference walls')
    footprint=occupied|floor_mask
    footprint_tree=cKDTree(np.column_stack([xx[footprint],yy[footprint]]))
    footprint_distance=footprint_tree.query(points[:,:2],workers=2)[0]
    outside_footprint=footprint_distance>AUDIT_POLICY['footprint_margin_m']
    outside_height=(points[:,2]<AUDIT_POLICY['minimum_z_m'])|(points[:,2]>AUDIT_POLICY['maximum_z_m'])
    floor_xy=np.column_stack([xx[floor_mask],yy[floor_mask]])
    ground=points[np.abs(points[:,2])<=.06]
    if len(ground)==0:raise ValueError('no measured ground points')
    ground_tree=cKDTree(ground[:,:2]);floor_dist=ground_tree.query(floor_xy,workers=2)[0]
    wall_points=points[(points[:,2]>.25)&(points[:,2]<2.85)]
    if len(wall_points)==0:raise ValueError('no measured wall points')
    best=np.full(len(wall_points),np.inf)
    for _,pos,size,rot in boxes:
        local=(wall_points[:,:2]-pos[:2])@rot;q=np.abs(local)-size[:2]
        sd=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
        best=np.minimum(best,np.abs(sd))
    wall_tree=cKDTree(wall_points)
    targets=[];wall_ids=[];diagnostic_targets={'wall_end_faces':[],'wall_lower_strip':[],'wall_upper_strip':[]}
    def inside(xy):
        ij=np.floor((xy-low)/res).astype(int)
        valid=(ij[:,0]>=0)&(ij[:,0]<xx.shape[1])&(ij[:,1]>=0)&(ij[:,1]<xx.shape[0])
        result=np.zeros(len(xy),bool);result[valid]=floor_mask[ij[valid,1],ij[valid,0]];return result
    for name,pos,size,rot in boxes:
        # These extra faces/heights are reported independently: the collision
        # approximation does not justify calling their score exact completeness.
        for axis in (0,1):
            for side in (-1,1):
                along=np.arange(-size[axis]+.025,size[axis],.05)
                face=np.zeros((len(along),2));face[:,axis]=along;face[:,1-axis]=side*size[1-axis]
                adjacent=face.copy();adjacent[:,1-axis]+=side*.12
                face=face[inside(adjacent@rot.T+pos[:2])]@rot.T+pos[:2]
                if not len(face):continue
                ranges={'wall_end_faces':np.arange(.3,2.71,.1)} if axis==1 else {}
                ranges.update(wall_lower_strip=np.array([.05,.15,.25]),wall_upper_strip=np.array(AUDIT_POLICY['top_strip_heights_m']))
                for label,z in ranges.items():
                    diagnostic_targets[label].append(np.column_stack([np.repeat(face,len(z),axis=0),np.tile(z,len(face))]))
        for side in (-1,1):
            along=np.arange(-size[0]+.1,size[0]-.05,.1)
            if not len(along):continue
            face=np.column_stack([along,np.full(len(along),side*size[1])])
            adjacent=face+np.array([0,side*.12]);visible=inside(adjacent@rot.T+pos[:2])
            face=face[visible]@rot.T+pos[:2]
            if not len(face):continue
            z=np.arange(.3,2.71,.1)
            xyz=np.column_stack([np.repeat(face,len(z),axis=0),np.tile(z,len(face))])
            targets.append(xyz);wall_ids.extend([name]*len(xyz))
    targets=np.concatenate(targets);surface_dist=wall_tree.query(targets,workers=2)[0]
    result={'pcd':str(path),'native_sha256':native_digest,'reference':str(xml),'reference_role':'offline_validation_only','reference_walls':len(boxes),'reference_floor_area_m2':len(floor_xy)*res*res,'alignment':'none; original world coordinates','floor_coverage':{str(t):float(np.mean(floor_dist<=t)) for t in [.05,.1,.15,.2]},'interior_wall_surface_coverage':{str(t):float(np.mean(surface_dist<=t)) for t in [.05,.1,.15,.2]},'wall_point_residual_m':{str(q):float(np.percentile(best,q)) for q in [50,90,95,99]},'wall_point_fraction_beyond':{str(t):float(np.mean(best>t)) for t in [.05,.1,.2]},'worst_walls':[]}
    wall_ids=np.array(wall_ids)
    for name in np.unique(wall_ids):
        d=surface_dist[wall_ids==name];result['worst_walls'].append({'wall':str(name),'target_count':len(d),'surface_coverage_10cm':float(np.mean(d<=.1)),'surface_coverage_15cm':float(np.mean(d<=.15)),'surface_coverage_20cm':float(np.mean(d<=.2)),'max_missing_distance_m':float(d.max())})
    result['worst_walls'].sort(key=lambda x:x['surface_coverage_10cm'])
    # Collision boxes omit real decorative protrusions of about 5..15 cm:
    # their residuals cannot certify absolute registration accuracy. Do not
    # mistake the collision model for a surveyed render mesh ground truth.
    result['reference_sha256']=reference_digest
    result['reference_wall_target_count']=len(targets)
    result['reference_floor_sampling']={'grid_resolution_m':res,'seed_xy_m':[0.,0.],'robot_center_seam_clearance_m':.32,'target_count':len(floor_xy)}
    result['audit_policy']=AUDIT_POLICY
    result['auditor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['pcd_reader_sha256']=hashlib.sha256((ROOT/'dev_ws/tools/pcd_quality_report.py').read_bytes()).hexdigest()
    result['all_point_spatial_support']={
        'points':len(points),'outside_footprint_points':int(outside_footprint.sum()),
        'outside_footprint_fraction':float(outside_footprint.mean()),
        'maximum_footprint_distance_m':float(footprint_distance.max()),
        'outside_height_points':int(outside_height.sum()),
        'clusters':spatial_outlier_clusters(points[outside_footprint],AUDIT_POLICY['outlier_cluster_voxel_m']),
        'cluster_gate':'diagnostic only; rendered surfaces outside collision seams require raw-data review',
    }
    all_tree=cKDTree(points)
    result['additional_surface_diagnostics']={}
    for label,chunks in diagnostic_targets.items():
        if not chunks:continue
        samples=np.concatenate(chunks);dist=all_tree.query(samples,workers=2)[0]
        result['additional_surface_diagnostics'][label]={'target_count':len(samples),'coverage_at_15cm':float(np.mean(dist<=.15)),'coverage_at_20cm':float(np.mean(dist<=.2)),'maximum_gap_m':float(dist.max()),'worst_target_xyz_m':samples[int(dist.argmax())].tolist(),'gating':label=='wall_upper_strip'}
    result['height_band_support']={}
    for label,lo,hi in [('below_ground',-.31,-.06),('floor',-.06,.06),('lower_wall',.06,.25),('main_wall',.25,2.85),('upper_wall',2.85,3.36)]:
        band=(points[:,2]>=lo)&(points[:,2]<hi)
        result['height_band_support'][label]={'points':int(band.sum()),'outside_footprint_points':int((band&outside_footprint).sum())}
    result['legacy_geometry_ignored_points']=int(((np.abs(points[:,2])>.06)&~((points[:,2]>.25)&(points[:,2]<2.85))).sum())
    result['limitations']=[
        'Collision boxes simplify the visible mesh; floor mask seals robot-inaccessible seams.',
        'Coverage uses nearest measured points at stated spatial tolerances, not exact surface completeness.',
        'No rigid alignment or reference-generated points are applied to the native PCD.',
        'These proxy checks cannot certify a perfect map or zero drift.'
        ,'Wall end faces and lower strips are diagnostic; the upper-strip gate checks observation support for the 3 m collision-wall proxy, not surveyed visible-mesh completeness.'
        ,'The all-point footprint test detects spatial anomalies at every height but is not a temporal drift measurement.'
    ]
    result['floor_largest_gap_radius_m']=float(floor_dist.max())
    result['floor_missing_area_beyond_10cm_m2']=float(np.count_nonzero(floor_dist>.1)*res*res)
    gaps=np.zeros(floor_mask.shape,bool);gaps[floor_mask]=floor_dist>.1
    labels,n=ndimage.label(gaps);clusters=[]
    for i in range(1,n+1):
        ij=np.argwhere(labels==i);d=ground_tree.query(np.column_stack([xx[labels==i],yy[labels==i]]),workers=2)[0]
        if len(ij)*res*res<.01:continue
        k=int(d.argmax());clusters.append({'area_m2':float(len(ij)*res*res),'largest_gap_radius_m':float(d[k]),'worst_xy_m':[float(xx[tuple(ij[k])]),float(yy[tuple(ij[k])])]})
    result['floor_gap_clusters']=sorted(clusters,key=lambda v:v['largest_gap_radius_m'],reverse=True)
    checks={
        'floor_coverage_at_10cm_ge_99_5pct':result['floor_coverage']['0.1']>=.995,
        'floor_coverage_at_20cm_ge_99_999pct':result['floor_coverage']['0.2']>=.99999,
        'largest_floor_gap_radius_le_25cm':result['floor_largest_gap_radius_m']<=.25,
        'wall_coverage_at_15cm_ge_99_5pct':result['interior_wall_surface_coverage']['0.15']>=.995,
        'wall_median_reference_residual_le_5mm':result['wall_point_residual_m']['50']<=.005,
        'wall_outliers_beyond_20cm_le_0_01pct':result['wall_point_fraction_beyond']['0.2']<=.0001,
        'all_height_footprint_outliers_beyond_35cm_le_0_01pct':float(outside_footprint.mean())<=AUDIT_POLICY['maximum_footprint_outlier_fraction'],
        'all_points_within_configured_height_bounds':not bool(outside_height.any()),
        'each_wall_coverage_at_20cm_ge_99_5pct':all(w['surface_coverage_20cm']>=AUDIT_POLICY['minimum_per_wall_coverage_at_20cm'] for w in result['worst_walls']),
        'each_wall_largest_gap_le_25cm':all(w['max_missing_distance_m']<=AUDIT_POLICY['maximum_per_wall_gap_m'] for w in result['worst_walls']),
        'top_strip_coverage_at_20cm_ge_99_5pct':result['additional_surface_diagnostics']['wall_upper_strip']['coverage_at_20cm']>=AUDIT_POLICY['minimum_top_strip_coverage_at_20cm'],
        'top_strip_largest_gap_le_25cm':result['additional_surface_diagnostics']['wall_upper_strip']['maximum_gap_m']<=AUDIT_POLICY['maximum_top_strip_gap_m'],
    }
    result['proxy_checks']=checks
    result['proxy_check_pass']=bool(all(checks.values()))
    result['quality_state']='PASS_PROXY_CHECKS' if result['proxy_check_pass'] else 'NEEDS_REVIEW'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=native_digest or hashlib.sha256(xml.read_bytes()).hexdigest()!=reference_digest:
        raise ValueError('native PCD or reference changed during audit')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    fig,axes=plt.subplots(1,2,figsize=(18,8))
    ax=axes[0];sample=wall_points[::max(1,len(wall_points)//150000)];ax.scatter(sample[:,0],sample[:,1],s=.1,c='steelblue',alpha=.5)
    for _,pos,size,rot in boxes:
        poly=np.array([[-size[0],-size[1]],[size[0],-size[1]],[size[0],size[1]],[-size[0],size[1]],[-size[0],-size[1]]])@rot.T+pos[:2];ax.plot(poly[:,0],poly[:,1],'k-',lw=.5)
    ax.set_title('Measured wall points vs reference geometry (black)')
    ax=axes[1];sc=ax.scatter(floor_xy[:,0],floor_xy[:,1],c=np.minimum(floor_dist,.5),s=1,cmap='turbo',vmin=0,vmax=.5);fig.colorbar(sc,ax=ax,label='distance to measured floor (m)');ax.set_title('Floor coverage gaps: red means missing > 0.5 m')
    for ax in axes:ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out.with_suffix('.png'),dpi=150);plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('pcd',type=Path);parser.add_argument('--xml',type=Path,default=ROOT/'runtime/src/robot_mujoco/zsibot_robots/xgb/unreal.xml');parser.add_argument('--output',required=True,type=Path);a=parser.parse_args();audit(a.pcd,a.xml,a.output)
