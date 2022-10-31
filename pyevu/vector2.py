from __future__ import annotations
from typing import Union, List, TypeVar
import math
import numpy as np
from .mathf import rad2deg

# TODO: Redo type hints to account for class inheritance in Vector3 as well.

V = TypeVar('V', bound='Vector2')
"""Must inherit from Vector2 class."""

class Vector2:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __str__(self: V) -> str:
        return f'{type(self).__name__}({self.x},{self.y})'
    
    def __repr__(self: V) -> str:
        return self.__str__()

    def __key(self: V) -> tuple:
        return tuple([self.__class__] + list(self.__dict__.values()))

    def __hash__(self: V):
        return hash(self.__key())

    def __eq__(self: V, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented

    def __neg__(self: V) -> V:
        return type(self)(x=-self.x, y=-self.y)

    def __add__(self: V, other: Union[float, int, V]) -> V:
        if issubclass(type(other), Vector2):
            return type(self)(x=self.x + other.x, y=self.y + other.y)
        elif type(other) in [float, int]:
            return type(self)(x=self.x + other, y=self.y + other)
        else:
            raise TypeError(f"Can't add {type(other).__name__} to {type(self).__name__}")

    def __radd__(self: V, other: Union[float, int, V]) -> V:
        return self.__add__(other)
    
    def __sub__(self: V, other: Union[float, int, V]) -> V:
        if type(other) in [float, int] or issubclass(type(other), Vector2):
            return self + (-other)
        else:
            raise TypeError(f"Can't subtract {type(other).__name__} from {type(self).__name__}")

    def __rsub__(self: V, other: Union[float, int, V]) -> V:
        return self.__neg__().__add__(other)
    
    def __mul__(self: V, other: Union[float, int]) -> V:
        if type(other) in [float, int]:
            return type(self)(x=self.x*other, y=self.y*other)
        else:
            raise TypeError(f"Can't multiply {type(self).__name__} with {type(other).__name__}")
    
    def __rmul__(self: V, other: Union[float, int]) -> V:
        return self.__mul__(other)

    def __truediv__(self: V, other: Union[float, int]) -> V:
        if type(other) in [float, int]:
            return self.__mul__(1/other)
        else:
            raise TypeError(f"Can't divide {type(other).__name__} from {type(self).__name__}")

    def __rtruediv__(self: V, other: Union[float, int]) -> V:
        if type(other) in [float, int]:
            return type(self)(x=1/self.x, y=1/self.y).__mul__(other)
        else:
            raise TypeError(f"Can't divide {type(self).__name__} from {type(other).__name__}")

    def __iter__(self: V):
        return iter([self.x, self.y])

    def Copy(self: V):
        return type(self).FromList(self.ToList())

    @classmethod
    @property
    def one(cls):
        return cls(1,1)
    
    @classmethod
    @property
    def zero(cls):
        return cls(0,0)
    
    @classmethod
    @property
    def left(cls):
        return cls(-1,0)
    
    @classmethod
    @property
    def right(cls):
        return cls(1,0)
    
    @classmethod
    @property
    def up(cls):
        return cls(0,1)
    
    @classmethod
    @property
    def down(cls):
        return cls(0,-1)

    @classmethod
    def Dot(cls, a: V, b: V) -> float:
        return a.x * b.x + a.y * b.y
    
    @classmethod
    def Dot2(cls, a: V) -> float:
        return cls.Dot(a, a)

    @classmethod
    def Cross(cls, a: V, b: V) -> float:
        return a.x * b.y - a.y * b.x
    
    @staticmethod
    def Project(a: V, b: V) -> V:
        b_norm = b.normalized
        scalarProjection = Vector2.Dot(a, b_norm)
        return scalarProjection * b_norm

    @property
    def magnitude(self) -> float:
        return (self.x**2 + self.y**2)**0.5
    
    @property
    def sqrMagnitude(self) -> float:
        return self.x**2 + self.y**2
    
    @property
    def normalized(self) -> V:
        if self.magnitude > 0:
            return self / self.magnitude
        else:
            return Vector2.zero


    @classmethod
    def Distance(cls, a: V, b: V) -> float:
        return (b-a).magnitude
    
    @classmethod
    def SqrDistance(cls, a: V, b: V) -> float:
        return (b-a).sqrMagnitude

    @classmethod
    def Angle(cls, a: V, b: V, deg: bool=False) -> float:
        aMag = a.magnitude
        bMag = b.magnitude
        if aMag == 0 or bMag == 0:
            return None
        ratio = cls.Dot(a,b) / (aMag * bMag)
        if abs(ratio) >= 1:
            angle = 0 if ratio > 0 else math.pi
            if deg:
                angle *= rad2deg
            return angle
        else:
            angle = math.acos(ratio)
            if deg:
                angle *= rad2deg
            return angle

    @classmethod
    def SignedAngle(cls, a: V, b: V, deg: bool=False) -> float:
        # Returns the signed angle in degrees between a and b.
        aMag = a.magnitude
        bMag = b.magnitude
        if aMag == 0 or bMag == 0:
            return None
        ratio = cls.Dot(a,b) / (aMag * bMag)
        if abs(ratio) >= 1:
            angle = 0 if ratio > 0 else math.pi
            if deg:
                angle *= rad2deg
            return angle
        else:
            angle = math.acos(ratio)
            if deg:
                angle *= rad2deg
            angle = math.copysign(angle, cls.Cross(a, b))
        return angle

    @staticmethod
    def Lerp(start: V, end: V, interpolationRatio: float) -> V:
        return start + (end - start) * interpolationRatio

    def ToList(self) -> List[float]:
        return [self.x, self.y]
    
    @classmethod
    def FromList(cls, vals: List[float]) -> V:
        if len(vals) != 2:
            raise Exception(f"Expected a list of length 2. Encountered length: {len(vals)}")
        return cls(vals[0], vals[1])
    
    def ToNumpy(self) -> np.ndarray:
        return np.array(self.ToList())

    @classmethod
    def FromNumpy(cls, arr: np.ndarray) -> V:
        return cls.FromList(arr.reshape(-1).tolist())

    def transpose(self) -> V:
        # Note: Since there are only two values, there is only one possible new order.
        # Therefore, it is not necessary to specify the order.
        return type(self)(x=self.y, y=self.x)

    def rotate(self, angle: float, deg: bool=True) -> V:
        # https://answers.unity.com/questions/661383/whats-the-most-efficient-way-to-rotate-a-vector2-o.html
        # Note: Rotates counter-clockwise.
        if deg:
            angle = math.radians(angle)
        sin = math.sin(angle)
        cos = math.cos(angle)
        tx = self.x
        ty = self.y
        return type(self)(
            x=(cos * tx) - (sin * ty),
            y=(sin * tx) + (cos * ty)
        )

    @property
    def mat2(self) -> np.ndarray:
        return self.ToNumpy().reshape(1,-1)
    
    @property
    def mat3(self) -> np.ndarray:
        return np.pad(self.mat2, [(0, 0), (0, 1)], mode='constant', constant_values=1)