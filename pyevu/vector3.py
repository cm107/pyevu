from __future__ import annotations
from typing import Union
import math
import numpy as np
from .mathf import rad2deg
from .random import Random
from .vector2 import Vector2

class Vector3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self) -> str:
        return f'Vector3({self.x},{self.y},{self.z})'
    
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
    
    def __iter__(self):
        return iter([self.x, self.y, self.z])

    def Copy(self) -> Vector3:
        return Vector3.FromList(self.ToList())

    @classmethod
    def RandomRange(
        cls, x: tuple[float, float], y: tuple[float, float], z: tuple[float, float]
    ) -> Vector3:
        return Vector3(
            x=Random.FloatRange(*x),
            y=Random.FloatRange(*y),
            z=Random.FloatRange(*z)
        )

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
    def Cross(cls, a: Vector3, b: Vector3) -> Vector3:
        return Vector3(
            x=a.y * b.z - a.z * b.y,
            y=-a.x * b.z + a.z * b.x,
            z=a.x * b.y - a.y * b.x
        )
    
    @staticmethod
    def Project(a: Vector3, b: Vector3) -> Vector3:
        b_norm = b.normalized
        scalarProjection = Vector3.Dot(a, b_norm)
        return scalarProjection * b_norm

    @property
    def magnitude(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)**0.5
    
    @property
    def sqrMagnitude(self) -> float:
        return self.x**2 + self.y**2 + self.z**2

    @property
    def normalized(self) -> Vector3:
        if self.magnitude > 0:
            return self / self.magnitude
        else:
            return Vector3.zero

    @classmethod
    def Distance(cls, a: Vector3, b: Vector3) -> float:
        return (b-a).magnitude
    
    @classmethod
    def SqrDistance(cls, a: Vector3, b: Vector3) -> float:
        return (b-a).sqrMagnitude

    @classmethod
    def Angle(cls, a: Vector3, b: Vector3, deg: bool=False) -> float:
        aMag = a.magnitude
        bMag = b.magnitude
        if aMag == 0 or bMag == 0:
            return None
        ratio = Vector3.Dot(a,b) / (aMag * bMag)
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
    def SignedAngle(cls, a: Vector3, b: Vector3, axis: Vector3, deg: bool=False) -> float:
        # Returns the signed angle in degrees between a and b.
        aMag = a.magnitude
        bMag = b.magnitude
        if aMag == 0 or bMag == 0:
            return None
        ratio = Vector3.Dot(a,b) / (aMag * bMag)
        if abs(ratio) >= 1:
            angle = 0 if ratio > 0 else math.pi
            if deg:
                angle *= rad2deg
            return angle
        else:
            angle = math.acos(ratio)
            if deg:
                angle *= rad2deg
            angle = math.copysign(angle, Vector3.Dot(axis, Vector3.Cross(a, b)))
            return angle

    @staticmethod
    def Lerp(start: Vector3, end: Vector3, interpolationRatio: float) -> Vector3:
        return start + (end - start) * interpolationRatio

    def ToList(self) -> list[float]:
        return [self.x, self.y, self.z]
    
    @staticmethod
    def FromList(vals: list[float]) -> Vector3:
        if len(vals) != 3:
            raise Exception(f"Expected a list of length 3. Encountered length: {len(vals)}")
        return Vector3(vals[0], vals[1], vals[2])
    
    def ToTuple(self) -> tuple[float]:
        return (self.x, self.y, self.z)
    
    @staticmethod
    def FromTuple(vals: tuple[float]) -> Vector3:
        if len(vals) != 3:
            raise Exception(f"Expected a tuple of length 3. Encountered length: {len(vals)}")
        return Vector3(vals[0], vals[1], vals[2])

    def ToNumpy(self) -> np.ndarray:
        return np.array(self.ToList())

    @staticmethod
    def FromNumpy(arr: np.ndarray) -> Vector3:
        return Vector3.FromList(arr.reshape(-1).tolist())

    def ToVector2(self) -> Vector2:
        return Vector2(self.x, self.y)
    
    @staticmethod
    def FromVector2(vec: Vector2) -> Vector3:
        return Vector3(vec.x, vec.y, 0)

    def transpose(self, order: str, inverse: bool=False) -> Vector3:
        # Example: vec3.transpose('zyx')
        assert len(order) == 3, f"order must be represented with 3 characters"
        expectedLetters = list('xyz')
        for letter in order:
            if letter not in expectedLetters:
                raise ValueError(f"Invalid character: {letter}. Expected one of the following: {expectedLetters}")
        for letter in expectedLetters:
            if letter not in list(order):
                raise ValueError(f"Character missing from order: {letter}. Received: {order}")
        currentValues = {'x': self.x, 'y': self.y, 'z': self.z}
        newValues = {}
        for i in range(len(expectedLetters)):
            if not inverse:
                newValues[expectedLetters[i]] = currentValues[list(order)[i]]
            else:
                newValues[list(order)[i]] = currentValues[expectedLetters[i]]
        return Vector3(x=newValues['x'], y=newValues['y'], z=newValues['z'])
    
    @property
    def mat3(self) -> np.ndarray:
        return self.ToNumpy().reshape(1,-1)
    
    @property
    def mat4(self) -> np.ndarray:
        return np.pad(self.mat3, [(0, 0), (0, 1)], mode='constant', constant_values=1)
    
    @property
    def translation_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, self.x],
            [0, 1, 0, self.y],
            [0, 0, 1, self.z],
            [0, 0, 0, 1]
        ])