from __future__ import annotations
from typing import List
from .vector3 import Vector3
from .interval import Interval

class BBox3D:
    def __init__(self, v0: Vector3, v1: Vector3):
        self.v0 = v0
        self.v1 = v1
    
    def __str__(self) -> str:
        return f"BBox3D({self.v0} ~ {self.v1})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
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
    
    def Contains(self, vertex: Vector3) -> bool:
        return self.ContainsX(vertex.x) and self.ContainsY(vertex.y) and self.ContainsZ(vertex.z)

    @classmethod
    def Union(cls, bbox0: BBox3D, bbox1: BBox3D) -> BBox3D:
        workingValues = BBox3D.WorkingValues()
        for point in [bbox0.v0, bbox0.v1, bbox1.v0, bbox1.v1]:
            workingValues.Update(point)
        return workingValues.ToBBox3D()
    
    @classmethod
    def Intersection(cls, bbox0: BBox3D, bbox1: BBox3D) -> BBox3D:
        xIntersection = Interval.Intersection(bbox0.xInterval, bbox1.xInterval)
        if xIntersection is None:
            return None
        yIntersection = Interval.Intersection(bbox0.yInterval, bbox1.yInterval)
        if yIntersection is None:
            return None
        zIntersection = Interval.Intersection(bbox0.zInterval, bbox1.zInterval)
        if zIntersection is None:
            return None
        return BBox3D(
            v0=Vector3(x=xIntersection.min, y=yIntersection.min, z=zIntersection.min),
            v1=Vector3(x=xIntersection.max, y=yIntersection.max, z=zIntersection.max)
        )
    
    @property
    def volume(self) -> float:
        return self.xInterval.length * self.yInterval.length * self.zInterval.length
