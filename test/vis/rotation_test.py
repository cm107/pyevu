from pyevu.vis.xyz_marker import XYZObject, XYZObjectScene
from pyevu import Transform, Quat, Vector3

def obj0_update(step: int, n_frames: int) -> Transform:
    angle = 360 * (step + 1) / n_frames
    return Transform(position=Vector3.right * 0.5, rotation=Quat.Euler(angle, 180, 0))

obj0 = XYZObject(update=obj0_update, label='obj0', s=0.5)

def obj1_update(step: int, n_frames: int) -> Transform:
    angle = 360 * (step + 1) / n_frames
    return Transform(position=Vector3.left * 0.5, rotation=Quat.Euler(angle, 0, 0))

obj1 = XYZObject(update=obj1_update, label='obj1', s=0.5)

scene = XYZObjectScene(
    obj_list=[obj0, obj1], n_frames=50,
    xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
)
scene.run()