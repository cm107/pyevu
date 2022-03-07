from .vector3 import Vector3
from .quaternion import Quaternion

class Transform:
    def __init__(self, position: Vector3, rotation: Quaternion):
        self.position = position
        self.rotation = rotation
    
    @property
    def forward(self) -> Vector3:
        return self.rotation * Vector3.forward

    @property
    def backward(self) -> Vector3:
        return self.rotation * Vector3.backward

    @property
    def left(self) -> Vector3:
        return self.rotation * Vector3.left

    @property
    def right(self) -> Vector3:
        return self.rotation * Vector3.right

    @property
    def up(self) -> Vector3:
        return self.rotation * Vector3.up

    @property
    def down(self) -> Vector3:
        return self.rotation * Vector3.down
