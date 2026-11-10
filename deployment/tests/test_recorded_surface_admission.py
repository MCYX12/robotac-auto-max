"""Regress actual missed-wall returns and isolated ghosts with production C++."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent

@pytest.fixture(scope='session')
def recorded_surface_probe(tmp_path_factory):
    build=tmp_path_factory.mktemp('recorded_surface_probe');binary=build/'probe'
    source=HERE.parents[1]/'dev_ws/src/robotac_mapping_baseline/src'
    subprocess.run(['g++','-O2','-std=c++17','-I'+str(source),'-I/usr/include/pcl-1.12',
                    '-I/usr/include/eigen3',str(HERE/'surface_recorded_probe.cpp'),
                    '-o',str(binary),'-lpcl_kdtree','-lpcl_common','-llz4'],check=True,
                   capture_output=True,text=True)
    return binary

@pytest.mark.parametrize('frame',range(9))
def test_recorded_surface_returns_and_ghosts(frame,tmp_path,recorded_surface_probe):
    fixture=HERE/'fixtures/surface_admission_recorded.npz'
    metadata=json.loads(fixture.with_suffix('.json').read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()==metadata['sha256']
    with np.load(fixture,allow_pickle=False) as data:
        a,b=data['offsets'][frame:frame+2];points=data['points'][a:b]
        select=data['query_frame']==frame;queries=data['query_index'][select];expected=data['expected'][select]
        assert len(queries)>0
        source=tmp_path/'measured.txt';np.savetxt(source,points,header=str(len(points)),comments='',fmt='%.9g')
        result=subprocess.run([str(recorded_surface_probe),str(source)],capture_output=True,text=True,check=True,timeout=15)
        keep=np.array([bool(int(v)) for v in result.stdout.split()]);assert len(keep)==len(points)
        assert np.array_equal(keep[queries],expected),metadata['frames'][frame]
