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

import numpy.typing as npt
from typing import Annotated, Literal, Generator

DType = TypeVar("DType", bound=np.generic)
ArrayNx1 = Annotated[npt.NDArray[DType], Literal["N", 1]]
ArrayNx2 = Annotated[npt.NDArray[DType], Literal["N", 2]]

class Vector2Arr:
    dtype = np.float64

    def __init__(self, x: ArrayNx1, y: ArrayNx1):
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
        return str(self.to_numpy())

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __neg__(self) -> Vector2Arr:
        return type(self)(x=-self.x, y=-self.y)

    def __add__(
        self: VA, other: Union[float, int, V, VA]
    ) -> VA:
        if issubclass(type(other), Vector2Arr):
            return type(self)(x=self.x + other.x, y=self.y + other.y)
        elif issubclass(type(other), Vector2):
            # return type(self)(x=self.x + other.x, y=self.y + other.y)
            return type(self).from_numpy(
                self.to_numpy() + other.ToNumpy()
            )
        elif type(other) in [float, int]:
            return type(self)(x=self.x + other, y=self.y + other)
        else:
            raise TypeError(f"Can't add {type(other).__name__} to {type(self).__name__}")

    def __radd__(self: VA, other: Union[float, int, V, VA]) -> VA:
        return self.__add__(other)
    
    def __sub__(self: VA, other: Union[float, int, V, VA]) -> VA:
        if (
            type(other) in [float, int]
            or issubclass(type(other), Vector2)
            or issubclass(type(other), Vector2Arr)
        ):
            return self + (-other)
        else:
            raise TypeError(f"Can't subtract {type(other).__name__} from {type(self).__name__}")

    def __rsub__(self: VA, other: Union[float, int, V, VA]) -> VA:
        return self.__neg__().__add__(other)
    
    def __mul__(self: VA, other: Union[float, int]) -> VA:
        if type(other) in [float, int]:
            return type(self)(x=self.x*other, y=self.y*other)
        else:
            raise TypeError(f"Can't multiply {type(self).__name__} with {type(other).__name__}")
    
    def __rmul__(self: VA, other: Union[float, int]) -> VA:
        return self.__mul__(other)

    def __truediv__(self: VA, other: Union[float, int]) -> VA:
        if type(other) in [float, int]:
            return self.__mul__(1/other)
        else:
            raise TypeError(f"Can't divide {type(other).__name__} from {type(self).__name__}")

    def __rtruediv__(self: VA, other: Union[float, int]) -> VA:
        if type(other) in [float, int]:
            return type(self)(x=1/self.x, y=1/self.y).__mul__(other)
        else:
            raise TypeError(f"Can't divide {type(self).__name__} from {type(other).__name__}")

    def __getitem__(self, idx: int) -> Vector2:
        if type(idx) is not int:
            raise TypeError
        return Vector2(float(self.x[idx].tolist()), float(self.y[idx].tolist()))

    def __iter__(self) -> Generator[Vector2]:
        for i in range(len(self)):
            yield self[i]

    def to_numpy(self) -> ArrayNx2:
        return np.vstack([self.x, self.y]).T.astype(type(self).dtype)

    @classmethod
    def from_numpy(cls, arr: ArrayNx2) -> Vector2Arr:
        return Vector2Arr(x=arr[:,0], y=arr[:,1])

    def to_vectors(self) -> list[Vector2]:
        return [Vector2(float(x), float(y)) for x, y in zip(self.x.tolist(), self.y.tolist())]

    @classmethod
    def from_vectors(cls, vectors: list[Vector2]) -> Vector2Arr:
        return Vector2Arr.from_numpy(np.array([list(v) for v in vectors]))

    @classmethod
    def Dot(cls, a: Union[VectorArrVar, V], b: Union[VectorArrVar, V]) -> ArrayNx1:
        # return a.x * b.x + a.y * b.y
        if issubclass(type(a), Vector2):
            assert type(b) is Vector2Arr
            a = np.tile(a.ToNumpy(), (len(b),1))
            a = Vector2Arr.from_numpy(a)
        if issubclass(type(b), Vector2):
            assert type(a) is Vector2Arr
            b = np.tile(b.ToNumpy(), (len(a),1))
            b = Vector2Arr.from_numpy(b)

        if type(a) is Vector2Arr:
            a = a.to_numpy()

        if type(b) is Vector2Arr:
            b = b.to_numpy()

        return (a * b).sum(axis=1)

    @classmethod
    def Cross(cls, a: VectorArrVar, b: VectorArrVar) -> ArrayNx1:
        # return a.x * b.y - a.y * b.x

        if issubclass(type(a), Vector2):
            assert type(b) is Vector2Arr
            a = np.tile(a.ToNumpy(), (len(b),1))
            a = Vector2Arr.from_numpy(a)
        if issubclass(type(b), Vector2):
            assert type(a) is Vector2Arr
            b = np.tile(b.ToNumpy(), (len(a),1))
            b = Vector2Arr.from_numpy(b)

        if type(a) is Vector2Arr:
            a = a.to_numpy()
        if type(b) is Vector2Arr:
            b = b.to_numpy()
        return a[:,0] * b[:,1] - a[:,1] * b[:,0]

    @property
    def magnitude(self) -> ArrayNx1:
        # return (self.x**2 + self.y**2)**0.5
        return np.linalg.norm(self.to_numpy(), axis=1)
    
    @property
    def sqrMagnitude(self) -> ArrayNx1:
        # return self.x**2 + self.y**2
        return (self.x * self.x) + (self.y * self.y)
    
    @property
    def normalized(self) -> Vector2Arr:
        # if self.magnitude > 0:
        #     return self / self.magnitude
        # else:
        #     return Vector2.zero

        np.seterr(invalid='ignore')

        a = self.to_numpy()
        mag = self.magnitude
        magTile = np.tile(mag, (2,1)).T
        result = np.true_divide(a, magTile)
        result = np.nan_to_num(result)
        return Vector2Arr.from_numpy(result)

    @classmethod
    def Project(cls, a: VectorArrVar, b: VectorArrVar) -> VectorArrVar:
        # b_norm = b.normalized
        # scalarProjection = Vector2.Dot(a, b_norm)
        # return scalarProjection * b_norm

        if type(a) is Vector2Arr:
            aVec = a
        else:
            aVec = Vector2Arr.from_numpy(a)

        if type(b) is Vector2Arr:
            bVec = b
        else:
            bVec = Vector2Arr.from_numpy(b)
        
        b_norm = bVec.normalized
        scalarProjection = Vector2Arr.Dot(aVec, b_norm)
        scalarProjectionTile = np.tile(scalarProjection, (2,1)).T
        result = scalarProjectionTile * b_norm.to_numpy()
        return Vector2Arr.from_numpy(result)

    @staticmethod
    def unit_test():
        aVecs = [
            Vector2(0,0),
            Vector2(1,2),
            Vector2(3,4),
            Vector2(5,6),
            Vector2(7,8),
            Vector2(9,10),
        ]
        bVecs = [
            Vector2(0,0),
            Vector2(11,12),
            Vector2(13,14),
            Vector2(15,16),
            Vector2(17,18),
            Vector2(19,20),
        ][::-1]
        aArr = Vector2Arr.from_vectors(aVecs)
        bArr = Vector2Arr.from_vectors(bVecs)

        assert (
            aArr.to_numpy() == Vector2Arr.from_numpy(aArr.to_numpy()).to_numpy()
        ).all()

        def same_vector(v0: Vector2, v1: Vector2, thresh: float=1e-5) -> bool:
            diff = v1 - v0
            return abs(diff.x) < thresh and abs(diff.y) < thresh

        dotArr = Vector2Arr.Dot(aArr, bArr)
        crossArr = Vector2Arr.Cross(aArr, bArr)
        aMag = aArr.magnitude; bMag = bArr.magnitude
        aSqrMag = aArr.sqrMagnitude; bSqrMag = bArr.sqrMagnitude
        aNorm = aArr.normalized; bNorm = bArr.normalized
        projArr = Vector2Arr.Project(aArr, bArr)
        for i, (a, b) in enumerate(zip(aVecs, bVecs)):
            assert dotArr[i] == Vector2.Dot(a, b)
            assert crossArr[i] == Vector2.Cross(a, b)
            assert aMag[i] == a.magnitude and bMag[i] == b.magnitude
            assert aSqrMag[i] == a.sqrMagnitude and bSqrMag[i] == b.sqrMagnitude
            
            an = aNorm.to_vectors()[i]; bn = bNorm.to_vectors()[i]
            assert same_vector(an, a.normalized), f"{an=}, {a.normalized=}"
            assert same_vector(bn, b.normalized), f"{bn=}, {b.normalized=}"

            projVec = projArr.to_vectors()
            assert same_vector(projVec[i], Vector2.Project(a, b)), f"{projVec[i]=}, {Vector2.Project(a, b)=}"

        negArr = -aArr
        for i, a in enumerate(aVecs):
            assert same_vector(negArr.to_vectors()[i], -a)

        addArr = aArr + 5
        for i, a in enumerate(aVecs):
            assert same_vector(addArr.to_vectors()[i], a + 5)
        
        addArr = aArr + Vector2(5,-12)
        for i, a in enumerate(aVecs):
            assert same_vector(addArr.to_vectors()[i], a + Vector2(5,-12)), \
                f"{addArr.to_vectors()[i]=}, {a + Vector2(5,-12)=}"

        addArr = aArr + bArr
        for i, (a, b) in enumerate(zip(aVecs, bVecs)):
            assert same_vector(addArr.to_vectors()[i], a + b)
        
        subArr = aArr - 5
        for i, a in enumerate(aVecs):
            assert same_vector(subArr.to_vectors()[i], a - 5)

        subArr = aArr - Vector2(5,-12)
        for i, a in enumerate(aVecs):
            assert same_vector(subArr.to_vectors()[i], a - Vector2(5,-12)), \
                f"{subArr.to_vectors()[i]=}, {a - Vector2(5,-12)=}"

        subArr = aArr - bArr
        for i, (a, b) in enumerate(zip(aVecs, bVecs)):
            assert same_vector(subArr.to_vectors()[i], a - b)

        mulArr = aArr * 5
        for i, a in enumerate(aVecs):
            assert same_vector(mulArr.to_vectors()[i], a * 5)
        
        divArr = aArr / 5
        for i, a in enumerate(aVecs):
            assert same_vector(divArr.to_vectors()[i], a / 5)

        p1 = np.array(
            [
                [ 50,  20],
                [100, 200],
                [100, 200],
                [-40,  20],
                [100, 200]
            ]
        )
        p0 = np.array(
            [
                [  100,   200],
                [-5000, -5000],
                [   50,    20],
                [  100,   200],
                [   50,    20]
            ]
        )
        print(Vector2Arr.from_numpy(p1) - Vector2Arr.from_numpy(p0))
        assert (
            (
                Vector2Arr.from_numpy(p1) - Vector2Arr.from_numpy(p0)
            ).to_numpy() == (p1 - p0)
        ).all()

        print('Pass')

from typing import Union
Vector2Arr = Union[Vector2Arr, ArrayNx2]
VA = TypeVar('VA', bound=Vector2Arr)