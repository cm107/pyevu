from pyevu import Vector3, Quat
import numpy as np
from scipy.spatial.transform import Rotation as R

"""
>>> x, y, z = 30, 50, 16 # Unity rotation order
>>> R.from_euler('zxy', [z,x,y], degrees=True).as_matrix() # python
"""

global_pos0 = Vector3.zero
local_pos0 = Vector3.zero
grv0 = Vector3(0,0,0)
lrv0 = Vector3(0,0,0)
local_rot_mat0 = R.from_euler('zxy', lrv0.transpose('zxy').ToList(), degrees=True).as_matrix()
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
gt0_q = Quat.from_transformation_matrix(gt0)

global_pos1 = Vector3(-1.98, 2.96, 3.96)
local_pos1 = Vector3(-1.98, 2.96, 3.96)
grv1 = Vector3(27.64419, 316.8064, 336.4617)
lrv1 = Vector3(27.64419, 316.8064, 336.4617)
local_rot_mat1 = R.from_euler('zxy', lrv1.transpose('zxy').ToList(), degrees=True).as_matrix()
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
gt1_q = Quat.from_transformation_matrix(gt1)

# go1 rotation: (27.64419, 316.8064, 336.4617), localRotation: (27.64419, 316.8064, 336.4617)
Quat.Euler(27.64419, 316.8064, 336.4617)

global_pos2 = Vector3(0.1340735, 0.8431807, 4.609821)
local_pos2 = Vector3(2.75, -1.34, 0.1199996)
grv2 = Vector3(32.03989, 308.1168, 66.14113)
lrv2 = Vector3(7.212612, 355.085, 93.70296)
local_rot_mat2 = R.from_euler('zxy', lrv2.transpose('zxy').ToList(), degrees=True).as_matrix()
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
gt2_q = Quat.from_transformation_matrix(gt2)

global_pos3 = Vector3(3.047737, 4.144675, 3.695593)
local_pos3 = Vector3(1.610007, -0.4680046, -4.173005)
grv3 = Vector3(81.31769, 357.0598, 195.0495)
lrv3 = Vector3(24.94642, 312.1511, 72.28976)
local_rot_mat3 = R.from_euler('zxy', lrv3.transpose('zxy').ToList(), degrees=True).as_matrix()
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
gt3_q = Quat.from_transformation_matrix(gt3)

pos = Vector3.zero
global_t = np.eye(4)
global_t[3, 0:3] = pos.ToNumpy()
for name, local_pos, local_rot_mat, gt, gt_q in zip(
    ['go0', 'go1', 'go2', 'go3'],
    [local_pos0, local_pos1, local_pos2, local_pos3],
    [local_rot_mat0, local_rot_mat1, local_rot_mat2, local_rot_mat3],
    [gt0, gt1, gt2, gt3],
    [gt0_q, gt1_q, gt2_q, gt3_q]
):
    # if name == 'go2':
    #     break

    local_t = local_rot_mat.copy()
    local_t = np.pad(local_t, [(0, 1), (0, 1)], mode='constant', constant_values=0)
    local_t[0:4, 3] = np.array(local_pos.ToList() + [1])
    
    # print(f"Before {global_t.round(2)}")
    # global_t = local_t @ global_t
    global_t = global_t @ local_t
    pos = Vector3.FromNumpy(global_t[0:3, 3])

    print(f"{name}: {pos}")
    print(f"{local_t.round(2)}")
    print(f"{global_t.round(2)}")
    print(f"{gt.round(2)}")
    # print(f"{global_t.round(2) == gt.round(2)}")
    # print(f"{gt_q.eulerAngles=}")