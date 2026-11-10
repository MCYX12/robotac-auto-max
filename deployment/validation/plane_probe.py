import numpy as np,json
from scipy.spatial.transform import Rotation,Slerp
from map_audit import ROOT,boxes_from_xml
from calibrate_timing import load
path=ROOT/'deployment/validation/timing_capture_headless_20261102/timing_capture_headless_20261102_0.db3'
d=load(path);od=d['odom'];t=d['times'];raw=d['clouds'];valid=(t>od[0,0]+.2)&(t<od[-1,0]-.1);t=t[valid];raw=raw[valid]
q=Slerp(od[:,0],Rotation.from_quat(od[:,4:8]))(t-.04);pos=np.column_stack([np.interp(t-.04,od[:,0],od[:,k]) for k in (1,2,3)])
xyz=np.einsum('nij,nkj->nki',q.as_matrix(),raw+np.array([0,0,.3]))+pos[:,None,:]
boxes=boxes_from_xml(ROOT/'runtime/src/robot_mujoco/zsibot_robots/xgb/unreal.xml')
for label,cloud in [('first_1s',xyz[t<t[0]+1].reshape(-1,3)),('all',xyz.reshape(-1,3))]:
 cloud=cloud[(cloud[:,2]>.4)&(cloud[:,2]<2.5)];ds=[]
 for _,p,s,r in boxes:
  loc=(cloud[:,:2]-p[:2])@r;z=np.abs(loc)-s[:2];dist=np.linalg.norm(np.maximum(z,0),axis=1)+np.minimum(z.max(axis=1),0);ds.append(np.abs(dist))
 ds=np.array(ds);ids=ds.argmin(axis=0);best=ds.min(axis=0)
 print(label,'points',len(best),'residual_percentiles',np.percentile(best,[50,90,95,99]).tolist())
 for idx,box in enumerate(boxes):
  sel=(ids==idx)
  if sel.sum()<20:continue
  _,p,s,r=box;loc=(cloud[sel,:2]-p[:2])@r
  print(box[0],int(sel.sum()),'residual_q',np.percentile(best[sel],[50,90,95]).round(4).tolist(),'normal_local_y_q',np.percentile(loc[:,1],[1,10,50,90,99]).round(4).tolist())
 print('OUTLIER_EXAMPLES',cloud[(best>.08)&(best<.2)][:12].round(4).tolist())
