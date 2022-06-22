from pyevu.transform_refactor import Transform
from pyevu import Vector3, Quat

t0 = Transform(position=Vector3(1,2,3), rotation=Quat.Euler(10,20,30))
t1 = Transform.FromTransformationMatrix(t0.worldTransformationMatrix)
assert t0.position == t1.position and t0.rotation == t1.rotation
assert t0.TransformPoint(Vector3.zero) == t0.position

print('Test Passed')