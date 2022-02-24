from __future__ import annotations
import math
from typing import Union, List
from .mathf import rad2deg

class Vector2:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
        return f'Vector2({self.x},{self.y})'
    
    def __repr__(self) -> str:
        return self.__str__()

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
    def normalized(self) -> Vector2:
        if self.magnitude > 0:
            return self / self.magnitude
        else:
            return Vector2.zero


    def Distance(cls, a: Vector2, b: Vector2) -> float:
        return (b-a).magnitude
    
    @classmethod
    def Angle(cls, a: Vector2, b: Vector2, deg: bool=False) -> float:
        if not deg:
            return math.acos(Vector2.Dot(a,b) / (a.magnitude * b.magnitude))
        else:
            return math.acos(Vector2.Dot(a,b) / (a.magnitude * b.magnitude)) * rad2deg

    def ToList(self) -> List[float]:
        return [self.x, self.y]
    
    @classmethod
    def FromList(self, vals: List[float]) -> Vector2:
        return Vector2(vals[0], vals[1])