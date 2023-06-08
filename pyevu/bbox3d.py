from __future__ import annotations
from typing import List, Union
from .vector3 import Vector3
from .interval import Interval

class BBox3D:
    def __init__(self, v0: Union[Vector3, tuple], v1: Union[Vector3, tuple]):
        def convert(value: Union[Vector3, tuple]) -> Vector3:
            if type(value) is Vector3:
                return value
            elif type(value) is tuple:
                return Vector3(*value)
            else:
                raise TypeError(f"Invalid type: {type(value).__name__}")
        
        self.v0 = convert(v0)
        self.v1 = convert(v1)
    
    def __str__(self) -> str:
        return f"BBox3D({self.v0} ~ {self.v1})"
    
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

    def __add__(self, other) -> BBox3D:
        if type(other) is Vector3:
            return BBox3D(v0=self.v0 + other, v1=self.v1 + other)
        else:
            raise TypeError
    
    def __sub__(self, other) -> BBox3D:
        if type(other) is Vector3:
            return BBox3D(v0=self.v0 - other, v1=self.v1 - other)
        else:
            raise TypeError

    def to_dict(self) -> dict:
        return dict(
            v0=list(self.v0),
            v1=list(self.v1)
        )
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> BBox3D:
        return BBox3D(
            v0=Vector3(*item_dict['v0']),
            v1=Vector3(*item_dict['v1'])
        )

    def Copy(self) -> BBox3D:
        return BBox3D(self.v0, self.v1)

    @property
    def center(self) -> Vector3:
        return 0.5 * (self.v0 + self.v1)

    @center.setter
    def center(self, value: Vector3):
        diffVector = value - self.center
        self.v0 += diffVector
        self.v1 += diffVector

    @property
    def shape(self) -> Vector3:
        return self.v1 - self.v0
    
    @shape.setter
    def shape(self, value: Vector3):
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
            self.zmin = None
            self.zmax = None
        
        @property
        def isNull(self) -> bool:
            return self.xmin is None \
                or self.xmax is None \
                or self.ymin is None \
                or self.ymax is None \
                or self.zmin is None \
                or self.zmax is None
        
        def Update(self, point: Vector3):
            if (self.xmin is None or point.x < self.xmin):
                self.xmin = point.x
            if (self.xmax is None or point.x > self.xmax):
                self.xmax = point.x
            if (self.ymin is None or point.y < self.ymin):
                self.ymin = point.y
            if (self.ymax is None or point.y > self.ymax):
                self.ymax = point.y
            if (self.zmin is None or point.z < self.zmin):
                self.zmin = point.z
            if (self.zmax is None or point.z > self.zmax):
                self.zmax = point.z
        
        def ToBBox3D(self) -> BBox3D:
            if (self.isNull):
                raise Exception("One of the working values are still null.")
            vmin = Vector3(x=self.xmin, y=self.ymin, z=self.zmin)
            vmax = Vector3(x=self.xmax, y=self.ymax, z=self.zmax)
            return BBox3D(v0=vmin, v1=vmax)

    @classmethod
    def FromVertices(cls, vertices: List[Vector3]) -> BBox3D:
        workingValues = BBox3D.WorkingValues()
        for vertex in vertices:
            workingValues.Update(vertex)
        return workingValues.ToBBox3D()
    
    @property
    def xInterval(self) -> Interval:
        return Interval(min=self.v0.x, max=self.v1.x)
    
    @property
    def yInterval(self) -> Interval:
        return Interval(min=self.v0.y, max=self.v1.y)

    @property
    def zInterval(self) -> Interval:
        return Interval(min=self.v0.z, max=self.v1.z)
    
    def ContainsX(self, val: float) -> bool:
        return self.xInterval.Contains(val)
    
    def ContainsY(self, val: float) -> bool:
        return self.yInterval.Contains(val)

    def ContainsZ(self, val: float) -> bool:
        return self.zInterval.Contains(val)
    
    def Contains(self, obj: Union[Vector3, BBox3D]) -> bool:
        if type(obj) is Vector3:
            return self.ContainsX(obj.x) and self.ContainsY(obj.y) and self.ContainsZ(obj.z)
        elif type(obj) is BBox3D:
            return self.Contains(obj.v0) and self.Contains(obj.v1)
        else:
            raise TypeError
    
    @classmethod
    def Union(cls, *args: BBox3D) -> BBox3D:
        workingValues = BBox3D.WorkingValues()
        for bbox in args:
            workingValues.Update(bbox.v0)
            workingValues.Update(bbox.v1)
        return workingValues.ToBBox3D()

    @classmethod
    def Intersection(cls, *args: BBox3D) -> BBox3D:
        xIntersection = Interval.Intersection(*[obj.xInterval for obj in args])
        if xIntersection is None:
            return None
        yIntersection = Interval.Intersection(*[obj.yInterval for obj in args])
        if yIntersection is None:
            return None
        zIntersection = Interval.Intersection(*[obj.zInterval for obj in args])
        if zIntersection is None:
            return None
        return BBox3D(
            v0=Vector3(x=xIntersection.min, y=yIntersection.min, z=zIntersection.min),
            v1=Vector3(x=xIntersection.max, y=yIntersection.max, z=zIntersection.max)
        )
    
    @property
    def volume(self) -> float:
        return self.xInterval.length * self.yInterval.length * self.zInterval.length

    def Clamp(self, vec: Union[Vector3, BBox3D]) -> Union[Vector3, BBox3D]:
        if type(vec) is Vector3:
            return Vector3(
                x=self.xInterval.Clamp(vec.x),
                y=self.yInterval.Clamp(vec.y),
                z=self.zInterval.Clamp(vec.z)
            )
        elif type(vec) is BBox3D:
            return BBox3D(
                v0=self.Clamp(vec.v0),
                v1=self.Clamp(vec.v1)
            )
        else:
            raise TypeError

    @staticmethod
    def IoU(bbox0: BBox3D, bbox1: BBox3D) -> float:
        """Intersection over Union (IoU)
        Refer to https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        """
        intersection = BBox3D.Intersection(bbox0, bbox1)
        if intersection is None:
            return 0
        else:
            overlap = intersection.volume
            union = bbox0.volume + bbox1.volume - overlap
            return overlap / union
