from __future__ import annotations
import math
from typing import Union, List
from .mathf import rad2deg

class Vector3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self) -> str:
        return f'Vector3({self.x},{self.y},{self.z})'
    
    def __repr__(self) -> str:
        return self.__str__()

    def __neg__(self) -> Vector3:
        return Vector3(x=-self.x, y=-self.y, z=-self.z)

    def __add__(self, other: Union[Vector3, float, int]) -> Vector3:
        if type(other) is Vector3:
            return Vector3(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)
        elif type(other) in [float, int]:
            return Vector3(x=self.x + other, y=self.y + other, z=self.z + other)
        else:
            raise TypeError(f"Can't add {type(other).__name__} to {type(self).__name__}")

    def __radd__(self, other: Union[Vector3, float, int]) -> Vector3:
        return self.__add__(other)
    
    def __sub__(self, other: Union[Vector3, float, int]) -> Vector3:
        if type(other) in [Vector3, float, int]:
            return self + (-other)
        else:
            raise TypeError(f"Can't subtract {type(other).__name__} from {type(self).__name__}")

    def __rsub__(self, other: Union[Vector3, float, int]) -> Vector3:
        return self.__neg__().__add__(other)
    
    def __mul__(self, other: Union[float, int]) -> Vector3:
        if type(other) in [float, int]:
            return Vector3(x=self.x*other, y=self.y*other, z=self.z*other)
        else:
            raise TypeError(f"Can't multiply {type(self).__name__} with {type(other).__name__}")
    
    def __rmul__(self, other: Union[float, int]) -> Vector3:
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, int]) -> Vector3:
        if type(other) in [float, int]:
            return self.__mul__(1/other)
        else:
            raise TypeError(f"Can't divide {type(other).__name__} from {type(self).__name__}")

    def __rtruediv__(self, other: Union[float, int]) -> Vector3:
        if type(other) in [float, int]:
            return Vector3(x=1/self.x, y=1/self.y, z=1/self.z).__mul__(other)
        else:
            raise TypeError(f"Can't divide {type(self).__name__} from {type(other).__name__}")

    @classmethod
    @property
    def one(cls) -> Vector3:
        return Vector3(1,1,1)
    
    @classmethod
    @property
    def zero(cls) -> Vector3:
        return Vector3(0,0,0)
    
    @classmethod
    @property
    def left(cls) -> Vector3:
        return Vector3(-1,0,0)
    
    @classmethod
    @property
    def right(cls) -> Vector3:
        return Vector3(1,0,0)
    
    @classmethod
    @property
    def up(cls) -> Vector3:
        return Vector3(0,1,0)
    
    @classmethod
    @property
    def down(cls) -> Vector3:
        return Vector3(0,-1,0)
    
    @classmethod
    @property
    def forward(cls) -> Vector3:
        return Vector3(0,0,1)

    @classmethod
    @property
    def backward(cls) -> Vector3:
        return Vector3(0,0,-1)

    @classmethod
    def Dot(cls, a: Vector3, b: Vector3) -> float:
        return a.x * b.x + a.y * b.y + a.z * b.z
    
    @classmethod
    def Dot2(cls, a: Vector3) -> float:
        return Vector3.Dot(a, a)

    @classmethod
    def Cross(cls, a: Vector3, b: Vector3) -> float:
        return Vector3(
            x=a.y * b.z - a.z * b.y,
            y=-a.x * b.z + a.z * b.x,
            z=a.x * b.y - a.y * b.x
        )
    
    @property
    def magnitude(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)**0.5
    
    @property
    def normalized(self) -> Vector3:
        return self / self.magnitude

    def Distance(cls, a: Vector3, b: Vector3) -> float:
        return (b-a).magnitude
    
    @classmethod
    def Angle(cls, a: Vector3, b: Vector3, deg: bool=False) -> float:
        if not deg:
            return math.acos(Vector3.Dot(a,b) / (a.magnitude * b.magnitude))
        else:
            return math.acos(Vector3.Dot(a,b) / (a.magnitude * b.magnitude)) * rad2deg

    def ToList(self) -> List[float]:
        return [self.x, self.y, self.z]
    
    @classmethod
    def FromList(self, vals: List[float]) -> Vector3:
        return Vector3(vals[0], vals[1], vals[2])