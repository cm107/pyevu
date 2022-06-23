from pyevu import GameObject, Transform, Vector3, Quat
from collections import OrderedDict

init_positions: OrderedDict[str, Vector3] = OrderedDict({
    'boomAxis': Vector3(0.0, 1.8, 2.1),
    'armAxis': Vector3(0.4, 4.0, 7.3),
    'bucketAxis': Vector3(0.3, 1.7, 5.6),
    'bucketTip': Vector3(0.2, 2.7, 4.8)
})

base = GameObject('base')
base.transform.SetPositionAndRotation(Vector3.zero, Quat.identity)

boomAxis = GameObject('boomAxis')
boomAxis.transform.parent = base.transform
boomAxis.transform.position = init_positions['boomAxis']

armAxis = GameObject('armAxis')
armAxis.transform.parent = boomAxis.transform
armAxis.transform.position = init_positions['armAxis']

bucketAxis = GameObject('bucketAxis')
bucketAxis.transform.parent = armAxis.transform
bucketAxis.transform.position = init_positions['bucketAxis']

bucketTip = GameObject('bucketTip')
bucketTip.transform.parent = bucketAxis.transform
bucketTip.transform.position = init_positions['bucketTip']

from pyevu.vis.xyz_marker import GameObjectXYZMarker, GameObjectMarkerScene

boomAxisMarker = GameObjectXYZMarker(boomAxis, s=0.5)
armAxisMarker = GameObjectXYZMarker(armAxis, s=0.5)
bucketAxisMarker = GameObjectXYZMarker(bucketAxis, s=0.5)
boomTipMarker = GameObjectXYZMarker(bucketTip, s=0.5)
markers = [boomAxisMarker, armAxisMarker, bucketAxisMarker, boomTipMarker]

def update_transforms(
    obj_dict: dict[str, GameObject], pressed_keys: list[str],
    step: int, n_frames: int
):
    # print(f"{pressed_keys=}")
    boomAxis = obj_dict['boomAxis']
    armAxis = obj_dict['armAxis']
    bucketAxis = obj_dict['bucketAxis']
    bucketTip = obj_dict['bucketTip']

    if 'z' in pressed_keys:
        boomAxis.transform.Rotate(Quat.Euler(0, -10, 0))
    elif 'x' in pressed_keys:
        boomAxis.transform.Rotate(Quat.Euler(0, 10, 0))
    if 'c' in pressed_keys:
        boomAxis.transform.Rotate(Quat.Euler(-10, 0, 0))
    elif 'v' in pressed_keys:
        boomAxis.transform.Rotate(Quat.Euler(10, 0, 0))
    if 'left' in pressed_keys:
        boomAxis.transform.Rotate(Quat.Euler(0, 0, -10))
    elif 'right' in pressed_keys:
        boomAxis.transform.Rotate(Quat.Euler(0, 0, 10))

    if 'b' in pressed_keys:
        armAxis.transform.Rotate(Quat.Euler(-10, 0, 0))
    elif 'n' in pressed_keys:
        armAxis.transform.Rotate(Quat.Euler(10, 0, 0))
    if 'm' in pressed_keys:
        bucketAxis.transform.Rotate(Quat.Euler(-10, 0, 0))
    elif ',' in pressed_keys:
        bucketAxis.transform.Rotate(Quat.Euler(10, 0, 0))


scene = GameObjectMarkerScene(
    obj_dict={marker.gameObject.name: marker for marker in markers},
    n_frames=200,
    xlim=(-10, 10), ylim=(0, 10), zlim=(-10, 10),
    update_transforms=update_transforms,
    draw_lines_to_children=True
)
scene.run()