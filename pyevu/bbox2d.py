from __future__ import annotations
from typing import List, Union, overload
import numpy as np
from .vector2 import Vector2
from .interval import Interval

class BBox2D:
    def __init__(
        self, v0: Union[Vector2, tuple], v1: Union[Vector2, tuple]
    ):
        def convert(value: Union[Vector2, tuple]) -> Vector2:
            if type(value) is Vector2:
                return value
            elif type(value) is tuple:
                return Vector2(*value)
            else:
                raise TypeError(f"Invalid type: {type(value).__name__}")
        
        self.v0 = convert(v0)
        self.v1 = convert(v1)
    
    def __str__(self) -> str:
        return f"BBox2D({self.v0} ~ {self.v1})"
    
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

    def __add__(self, other) -> BBox2D:
        if type(other) is Vector2:
            return BBox2D(v0=self.v0 + other, v1=self.v1 + other)
        else:
            raise TypeError
    
    def __sub__(self, other) -> BBox2D:
        if type(other) is Vector2:
            return BBox2D(v0=self.v0 - other, v1=self.v1 - other)
        else:
            raise TypeError

    def Copy(self) -> BBox2D:
        return BBox2D(self.v0, self.v1)

    def to_dict(self) -> dict:
        return dict(
            v0=list(self.v0),
            v1=list(self.v1)
        )
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> BBox2D:
        return BBox2D(
            v0=Vector2(*item_dict['v0']),
            v1=Vector2(*item_dict['v1'])
        )

    @property
    def center(self) -> Vector2:
        return 0.5 * (self.v0 + self.v1)

    @center.setter
    def center(self, value: Vector2):
        diffVector = value - self.center
        self.v0 += diffVector
        self.v1 += diffVector

    @property
    def shape(self) -> Vector2:
        return self.v1 - self.v0
    
    @shape.setter
    def shape(self, value: Vector2):
        shape = self.shape
        shape_diff = value - shape
        half_shape_diff = 0.5 * shape_diff
        self.v0 -= half_shape_diff
        self.v1 += half_shape_diff

    class WorkingValues:
        def __init__(self):
            self.xmin = None
            self.xmax = None
            self.ymin = None
            self.ymax = None
        
        @property
        def isNull(self) -> bool:
            return self.xmin is None \
                or self.xmax is None \
                or self.ymin is None \
                or self.ymax is None
        
        def Update(self, point: Vector2):
            if (self.xmin is None or point.x < self.xmin):
                self.xmin = point.x
            if (self.xmax is None or point.x > self.xmax):
                self.xmax = point.x
            if (self.ymin is None or point.y < self.ymin):
                self.ymin = point.y
            if (self.ymax is None or point.y > self.ymax):
                self.ymax = point.y
        
        def ToBBox2D(self) -> BBox2D:
            if (self.isNull):
                raise Exception("One of the working values are still null.")
            vmin = Vector2(x=self.xmin, y=self.ymin)
            vmax = Vector2(x=self.xmax, y=self.ymax)
            return BBox2D(v0=vmin, v1=vmax)

    @classmethod
    def FromVertices(cls, vertices: List[Vector2]) -> BBox2D:
        workingValues = BBox2D.WorkingValues()
        for vertex in vertices:
            workingValues.Update(vertex)
        return workingValues.ToBBox2D()
    
    @property
    def xInterval(self) -> Interval:
        return Interval(min=self.v0.x, max=self.v1.x)
    
    @property
    def yInterval(self) -> Interval:
        return Interval(min=self.v0.y, max=self.v1.y)
    
    def ContainsX(self, val: float) -> bool:
        return self.xInterval.Contains(val)
    
    def ContainsY(self, val: float) -> bool:
        return self.yInterval.Contains(val)
    
    def Contains(self, obj: Union[Vector2, BBox2D]) -> bool:
        if type(obj) is Vector2:
            return self.ContainsX(obj.x) and self.ContainsY(obj.y)
        elif type(obj) is BBox2D:
            return self.Contains(obj.v0) and self.Contains(obj.v1)
        else:
            raise TypeError

    @classmethod
    def Union(cls, *args: BBox2D) -> BBox2D:
        workingValues = BBox2D.WorkingValues()
        for bbox in args:
            workingValues.Update(bbox.v0)
            workingValues.Update(bbox.v1)
        return workingValues.ToBBox2D()
    
    @classmethod
    def Intersection(cls, *args: BBox2D) -> BBox2D:
        xIntersection = Interval.Intersection(*[obj.xInterval for obj in args])
        if xIntersection is None:
            return None
        yIntersection = Interval.Intersection(*[obj.yInterval for obj in args])
        if yIntersection is None:
            return None
        return BBox2D(
            v0=Vector2(x=xIntersection.min, y=yIntersection.min),
            v1=Vector2(x=xIntersection.max, y=yIntersection.max)
        )

    @property
    def area(self) -> float:
        return self.xInterval.length * self.yInterval.length

    @overload
    def Clamp(self, vec: Vector2) -> Vector2: ...

    @overload
    def Clamp(self, vec: BBox2D) -> BBox2D: ...

    def Clamp(self, vec: Union[Vector2, BBox2D]) -> Union[Vector2, BBox2D]:
        if type(vec) is Vector2:
            return Vector2(
                x=self.xInterval.Clamp(vec.x),
                y=self.yInterval.Clamp(vec.y)
            )
        elif type(vec) is BBox2D:
            return BBox2D(
                v0=self.Clamp(vec.v0),
                v1=self.Clamp(vec.v1)
            )
        else:
            raise TypeError

    @staticmethod
    def IoU(bbox0: BBox2D, bbox1: BBox2D) -> float:
        """Intersection over Union (IoU)
        Refer to https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        """
        intersection = BBox2D.Intersection(bbox0, bbox1)
        if intersection is None:
            return 0
        else:
            overlap = intersection.area
            union = bbox0.area + bbox1.area - overlap
            return overlap / union

    def crop_image(self, img: np.ndarray) -> np.ndarray:
        xmin, ymin = [int(val) for val in list(self.v0)]
        xmax, ymax = [int(val) for val in list(self.v1)]
        return img[ymin:ymax, xmin:xmax, :]

    def flatten(self) -> tuple[float, float, float, float]:
        return tuple(list(self.v0) + list(self.v1))