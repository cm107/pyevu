from __future__ import annotations
from typing import TypeVar
import copy
import numpy as np
from .vector2 import Vector2
from .line2 import Line2
from .bbox2d import BBox2D

class Quad2:
    def __init__(
        self,
        p0: Vector2, p1: Vector2,
        p2: Vector2, p3: Vector2
    ):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
    
    def __str__(self) -> str:
        paramStr = ','.join([str(list(p)) for p in self])
        return f'{type(self).__name__}({paramStr})'

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

    def __iter__(self):
        return iter([self.p0, self.p1, self.p2, self.p3])

    def __add__(self, other):
        if issubclass(type(other), Vector2):
            return type(self)(*[p + other for p in self])
        else:
            raise TypeError
    
    def __sub__(self, other):
        if issubclass(type(other), Vector2):
            return type(self)(*[p - other for p in self])
        else:
            raise TypeError

    def copy(self):
        return copy.deepcopy(self)

    def to_numpy(self) -> np.ndarray:
        return np.array(
            list(self.p0) + list(self.p1)
            + list(self.p2) + list(self.p3)
        )

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Q:
        return cls.from_list(arr.reshape(-1, 2).tolist())
    
    @property
    def centroid(self) -> Vector2:
        return (self.p0 + self.p1 + self.p2 + self.p3) / 4
    
    @property
    def lines(self) -> tuple[Line2, Line2, Line2, Line2]:
        return (
            Line2(self.p0, self.p1),
            Line2(self.p1, self.p2),
            Line2(self.p2, self.p3),
            Line2(self.p3, self.p0)
        )
    
    @property
    def bbox2d(self) -> BBox2D:
        return BBox2D.FromVertices(list(self))
    
    @classmethod
    def from_bbox2d(cls, bbox: BBox2D) -> Quad2:
        return Quad2(
            Vector2(bbox.v0.x, bbox.v1.y),
            Vector2(bbox.v1.x, bbox.v1.y),
            Vector2(bbox.v1.x, bbox.v0.y),
            Vector2(bbox.v0.x, bbox.v0.y)
        )

    def to_points(self) -> list[Vector2]:
        return [self.p0, self.p1, self.p2, self.p3]

    @classmethod
    def from_points(cls, points: list[Vector2]) -> Quad2:
        if type(points) is list:
            return Quad2(*points)
        elif type(points) is tuple:
            return Quad2(*list(points))
        else:
            raise TypeError

    def to_list(self) -> list[list[float]]:
        return [list(p) for p in self]

    @classmethod
    def from_list(cls, vals: list[list[float]]) -> Quad2:
        return Quad2(*[Vector2(*val) for val in vals])

Q = TypeVar('Q', bound=Quad2)