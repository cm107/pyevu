from pyevu import Quat, Vector3
from scipy.spatial.transform import Rotation as R

euler = Vector3(10, 20, 30)
assert (Quat.EulerVector(euler, order='zxy').rotation_matrix.round(2) == R.from_euler('zxy', euler.transpose('zxy').ToList(), degrees=True).as_matrix().round(2)).all()

for euler in [
    Vector3(10, 20, 30), Vector3(20, 20, 30), Vector3(30, 20, 30), Vector3(45, 20, 30), Vector3(80, 20, 30), Vector3(85, 20, 30), Vector3(89, 20, 30), Vector3(91, 20, 30),
    Vector3(10, 89, 30),
    Vector3(10, 20, 89),
    Vector3(10, 20, 120)
]:
    for order in ['xyz', 'yzx', 'zxy', 'zyx', 'xzy', 'yxz']:
        sq0 = euler.ToNumpy()
        sq1 = R.from_euler(order, euler.ToNumpy().tolist(), degrees=True).as_euler(order, degrees=True)
        assert (sq0.round(2) == sq1.round(2)).all(), f"scipy fail at {euler=}, {order=}. {sq1=}"

        euler0 = Quat.EulerVector(euler, deg=True, order=order).GetEulerAngles(deg=True, order=order)
        # print(f"{Quat.EulerVector(euler=euler, order=order).ToNumpy()=}")
        # print(f"{Quat.EulerVector(euler=euler0, order=order).ToNumpy()=}")
        # Note: q is the same as -q
        # https://math.stackexchange.com/questions/2016282/negative-quaternion
        assert (Quat.EulerVector(euler=euler, order=order).ToNumpy().round(2) == Quat.EulerVector(euler=euler0, order=order).ToNumpy().round(2)).all() \
            or (Quat.EulerVector(euler=euler, order=order).ToNumpy().round(2) == -Quat.EulerVector(euler=euler0, order=order).ToNumpy().round(2)).all(), \
            f"Failed at {euler=}, {order=}. {euler0=}"

# Look Rotation Test
def vec3_match(value: Vector3, expected: Vector3, decimals: int=2) -> bool:
    return (value.ToNumpy().round(decimals) == expected.ToNumpy().round(decimals)).all()

def euler_match(value: Vector3, expected: Vector3, decimals: int=2) -> bool:
    def standardize(v: float) -> float:
        v %= 360
        if v < 0:
            v += 360
        return v

    value = Vector3.FromList([standardize(v) for v in value.ToList()])
    expected = Vector3.FromList([standardize(v) for v in expected.ToList()])
    return vec3_match(value=value, expected=expected, decimals=decimals)

assert euler_match(
    Quat.LookRotation(Vector3.forward, Vector3.up).eulerAngles,
    Vector3(0, 0, 0)
), f"{Quat.LookRotation(Vector3.forward, Vector3.up).eulerAngles=}"
assert euler_match(
    Quat.LookRotation(Vector3.right, Vector3.up).eulerAngles,
    Vector3(0, 90, 0)
), f"{Quat.LookRotation(Vector3.right, Vector3.up).eulerAngles=}"
assert euler_match(
    Quat.LookRotation(Vector3.left, Vector3.forward).eulerAngles,
    Vector3(0, 270, 270)
), f"{Quat.LookRotation(Vector3.left, Vector3.forward).eulerAngles=}"

# # FromToRotation Test
# assert (
#     Quat.FromToRotation(Vector3.up, Vector3.right).eulerAngles.ToNumpy().round(2)
#     == Vector3(0, 0, -90).ToNumpy().round(2)
# ).all()
# assert (
#     Quat.FromToRotation(Vector3.up, Vector3.right + Vector3.up).eulerAngles.ToNumpy().round(2)
#     == Vector3(0, 0, -45).ToNumpy().round(2)
# ).all()

print("Passed Test")