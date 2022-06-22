from pyevu import Vector3, Quat, GameObject
import numpy as np
from scipy.spatial.transform import Rotation as R

global_pos0 = Vector3.zero
local_pos0 = Vector3.zero
grv0 = Vector3(0,0,0)
lrv0 = Vector3(0,0,0)
# local_rot_mat0 = R.from_euler('zxy', lrv0.transpose('zxy').ToList(), degrees=True).as_matrix()
# local_rot_mat0 = Quat.EulerVector(lrv0, order='zxy').rotation_matrix
go0 = GameObject(name='go0')
go0.transform.parent = None
go0.transform.localPosition = local_pos0
go0.transform.localRotation = Quat.EulerVector(lrv0)
gt0_str = """
1.00000	0.00000	0.00000	0.00000
0.00000	1.00000	0.00000	0.00000
0.00000	0.00000	1.00000	0.00000
0.00000	0.00000	0.00000	1.00000
"""
gt0 = np.array(
    [
        [float(val) for val in row.split('\t')]
        for row in gt0_str.split('\n')
        if len(row) > 0
    ]
)

global_pos1 = Vector3(-1.98, 2.96, 3.96)
local_pos1 = Vector3(-1.98, 2.96, 3.96)
grv1 = Vector3(27.64419, 316.8064, 336.4617)
lrv1 = Vector3(27.64419, 316.8064, 336.4617)
# local_rot_mat1 = R.from_euler('zxy', lrv1.transpose('zxy').ToList(), degrees=True).as_matrix()
# local_rot_mat1 = Quat.EulerVector(lrv1, order='zxy').rotation_matrix
go1 = GameObject(name='go1')
go1.transform.parent = go0.transform
go1.transform.localPosition = local_pos1
go1.transform.localRotation = Quat.EulerVector(lrv1)
gt1_str = """
0.79521	0.00000	-0.60633	-1.98000
-0.35377	0.81214	-0.46398	2.96000
0.49242	0.58347	0.64582	3.96000
0.00000	0.00000	0.00000	1.00000
"""
gt1 = np.array(
    [
        [float(val) for val in row.split('\t')]
        for row in gt1_str.split('\n')
        if len(row) > 0
    ]
)

# go1 rotation: (27.64419, 316.8064, 336.4617), localRotation: (27.64419, 316.8064, 336.4617)
Quat.Euler(27.64419, 316.8064, 336.4617)

global_pos2 = Vector3(0.1340735, 0.8431807, 4.609821)
local_pos2 = Vector3(2.75, -1.34, 0.1199996)
grv2 = Vector3(32.03989, 308.1168, 66.14113)
lrv2 = Vector3(7.212612, 355.085, 93.70296)
# local_rot_mat2 = R.from_euler('zxy', lrv2.transpose('zxy').ToList(), degrees=True).as_matrix()
# local_rot_mat2 = Quat.EulerVector(lrv2, order='zxy').rotation_matrix
go2 = GameObject(name='go2')
go2.transform.parent = go1.transform
go2.transform.localPosition = local_pos2
go2.transform.localRotation = Quat.EulerVector(lrv2)
gt2_str = """
-0.13204	-0.73334	-0.66691	0.13407
0.77524	0.34287	-0.53051	0.84318
0.61771	-0.58707	0.52324	4.60982
0.00000	0.00000	0.00000	1.00000
"""
gt2 = np.array(
    [
        [float(val) for val in row.split('\t')]
        for row in gt2_str.split('\n')
        if len(row) > 0
    ]
)

global_pos3 = Vector3(3.047737, 4.144675, 3.695593)
local_pos3 = Vector3(1.610007, -0.4680046, -4.173005)
grv3 = Vector3(81.31769, 357.0598, 195.0495)
lrv3 = Vector3(24.94642, 312.1511, 72.28976)
# local_rot_mat3 = R.from_euler('zxy', lrv3.transpose('zxy').ToList(), degrees=True).as_matrix()
# local_rot_mat3 = Quat.EulerVector(lrv3, order='zxy').rotation_matrix
go3 = GameObject(name='go3')
go3.transform.parent = go2.transform
go3.transform.localPosition = local_pos3
go3.transform.localRotation = Quat.EulerVector(lrv3)
gt3_str = """
-0.95127	0.30828	-0.00774	3.04774
-0.03920	-0.14578	-0.98854	4.14468
-0.30587	-0.94006	0.15076	3.69559
0.00000	0.00000	0.00000	1.00000
"""
gt3 = np.array(
    [
        [float(val) for val in row.split('\t')]
        for row in gt3_str.split('\n')
        if len(row) > 0
    ]
)

pos = Vector3.zero
global_t = np.eye(4)
global_t[3, 0:3] = pos.ToNumpy()
for go, gt in zip(
    [go0, go1, go2, go3],
    [gt0, gt1, gt2, gt3],
):
    print(f"{go.name}: {go.transform.position}")
    go.transform.PrintHierarchy()
    print(f"{go.transform.worldTransformationMatrix.round(2)}")
    print(f"{gt.round(2)}")