from pyevu.transform import Transform
from pyevu import Vector3, Quat

t0 = Transform(position=Vector3(1,2,3), rotation=Quat.Euler(10,20,30))
t1 = Transform.FromTransformationMatrix(t0.worldTransformationMatrix)
assert t0.position == t1.position and t0.rotation == t1.rotation
assert t0.TransformPoint(Vector3.zero) == t0.position

t0 = Transform(position=Vector3.zero, rotation=Quat.identity)
t0.Rotate(Quat.Euler(55,25,25))
assert (
    t0.rotation.eulerAngles.ToNumpy().round(2)
    == Vector3(55,25,25).ToNumpy().round(2)
).all()

t0 = Transform(position=Vector3(1,2,3), rotation=Quat.identity)
t1 = Transform(
    position=t0.position + Vector3.right + Vector3.forward,
    rotation=Quat.identity
)
t0.LookAt(t1, worldUp=Vector3.up)
assert (
    t0.forward.ToNumpy().round(2)
    == (Vector3.right + Vector3.forward).normalized.ToNumpy().round(2)
).all()
t2 = Transform(
    position=t0.position + Vector3.right,
    rotation=Quat.identity
)
t0.LookAt(t2, worldUp=Vector3.up)
assert (
    t0.forward.ToNumpy().round(2)
    == (Vector3.right).normalized.ToNumpy().round(2)
).all()
t3 = Transform(
    position=t0.position + Vector3.right + Vector3.forward + Vector3.up,
    rotation=Quat.identity
)
t0.LookAt(t3, worldUp=Vector3.up)
assert (
    t0.forward.ToNumpy().round(2)
    == (Vector3.right + Vector3.forward + Vector3.up).normalized.ToNumpy().round(2)
).all()

print('Test Passed')