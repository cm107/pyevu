from __future__ import annotations
from typing import Union, List
import math
import numpy as np
from .mathf import rad2deg

class Vector2:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
        return f'Vector2({self.x},{self.y})'
    
    def __repr__(self) -> str:
        return self.__str__()

    def __key(self) -> tuple:
        return tuple([self.__class__] + list(self.__dict__.values()))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented

    def __neg__(self) -> Vector2:
        return Vector2(x=-self.x, y=-self.y)

    def __add__(self, other: Union[Vector2, float, int]) -> Vector2:
        if type(other) is Vector2:
            return Vector2(x=self.x + other.x, y=self.y + other.y)
        elif type(other) in [float, int]:
            return Vector2(x=self.x + other, y=self.y + other)
        else:
            raise TypeError(f"Can't add {type(other).__name__} to {type(self).__name__}")

    def __radd__(self, other: Union[Vector2, float, int]) -> Vector2:
        return self.__add__(other)
    
    def __sub__(self, other: Union[Vector2, float, int]) -> Vector2:
        if type(other) in [Vector2, float, int]:
            return self + (-other)
        else:
            raise TypeError(f"Can't subtract {type(other).__name__} from {type(self).__name__}")

    def __rsub__(self, other: Union[Vector2, float, int]) -> Vector2:
        return self.__neg__().__add__(other)
    
    def __mul__(self, other: Union[float, int]) -> Vector2:
        if type(other) in [float, int]:
            return Vector2(x=self.x*other, y=self.y*other)
        else:
            raise TypeError(f"Can't multiply {type(self).__name__} with {type(other).__name__}")
    
    def __rmul__(self, other: Union[float, int]) -> Vector2:
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, int]) -> Vector2:
        if type(other) in [float, int]:
            return self.__mul__(1/other)
        else:
            raise TypeError(f"Can't divide {type(other).__name__} from {type(self).__name__}")

    def __rtruediv__(self, other: Union[float, int]) -> Vector2:
        if type(other) in [float, int]:
            return Vector2(x=1/self.x, y=1/self.y).__mul__(other)
        else:
            raise TypeError(f"Can't divide {type(self).__name__} from {type(other).__name__}")

    def Copy(self) -> Vector2:
        return Vector2.FromList(self.ToList())

    @classmethod
    @property
    def one(cls) -> Vector2:
        return Vector2(1,1)
    
    @classmethod
    @property
    def zero(cls) -> Vector2:
        return Vector2(0,0)
    
    @classmethod
    @property
    def left(cls) -> Vector2:
        return Vector2(-1,0)
    
    @classmethod
    @property
    def right(cls) -> Vector2:
        return Vector2(1,0)
    
    @classmethod
    @property
    def up(cls) -> Vector2:
        return Vector2(0,1)
    
    @classmethod
    @property
    def down(cls) -> Vector2:
        return Vector2(0,-1)

    @classmethod
    def Dot(cls, a: Vector2, b: Vector2) -> float:
        return a.x * b.x + a.y * b.y
    
    @classmethod
    def Dot2(cls, a: Vector2) -> float:
        return Vector2.Dot(a, a)

    @classmethod
    def Cross(cls, a: Vector2, b: Vector2) -> float:
        return a.x * b.y - a.y * b.x
    
    @property
    def magnitude(self) -> float:
        return (self.x**2 + self.y**2)**0.5
    
    @property
    def sqrMagnitude(self) -> float:
        return self.x**2 + self.y**2
    
    @property
    def normalized(self) -> Vector2:
        if self.magnitude > 0:
            return self / self.magnitude
        else:
            return Vector2.zero


    @classmethod
    def Distance(cls, a: Vector2, b: Vector2) -> float:
        return (b-a).magnitude
    
    @classmethod
    def SqrDistance(cls, a: Vector2, b: Vector2) -> float:
        return (b-a).sqrMagnitude

    @classmethod
    def Angle(cls, a: Vector2, b: Vector2, deg: bool=False) -> float:
        if not deg:
            return math.acos(Vector2.Dot(a,b) / (a.magnitude * b.magnitude))
        else:
            return math.acos(Vector2.Dot(a,b) / (a.magnitude * b.magnitude)) * rad2deg

    @classmethod
    def SignedAngle(cls, a: Vector2, b: Vector2, deg: bool=False) -> float:
        # Returns the signed angle in degrees between a and b.
        angle = Vector2.Angle(a, b, deg=deg)
        angle = math.copysign(angle, Vector2.Cross(a, b))
        return angle

    def ToList(self) -> List[float]:
        return [self.x, self.y]
    
    @classmethod
    def FromList(self, vals: List[float]) -> Vector2:
        if len(vals) != 2:
            raise Exception(f"Expected a list of length 2. Encountered length: {len(vals)}")
        return Vector2(vals[0], vals[1])
    
    def ToNumpy(self) -> np.ndarray:
        return np.array(self.ToList())

    @classmethod
    def FromNumpy(cls, arr: np.ndarray) -> Vector2:
        return Vector2.FromList(arr.reshape(-1).tolist())

    def transpose(self) -> Vector2:
        # Note: Since there are only two values, there is only one possible new order.
        # Therefore, it is not necessary to specify the order.
        return Vector2(x=self.y, y=self.x)

    def rotate(self, angle: float, deg: bool=True) -> Vector2:
        # https://answers.unity.com/questions/661383/whats-the-most-efficient-way-to-rotate-a-vector2-o.html
        # Note: Rotates counter-clockwise.
        if deg:
            angle = math.radians(angle)
        sin = math.sin(angle)
        cos = math.cos(angle)
        tx = self.x
        ty = self.y
        return Vector2(
            x=(cos * tx) - (sin * ty),
            y=(sin * tx) + (cos * ty)
        )

    @property
    def mat2(self) -> np.ndarray:
        return self.ToNumpy().reshape(1,-1)
    
    @property
    def mat3(self) -> np.ndarray:
        return np.pad(self.mat2, [(0, 0), (0, 1)], mode='constant', constant_values=1)