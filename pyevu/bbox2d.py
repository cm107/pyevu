from __future__ import annotations
from typing import List, Union
from .vector2 import Vector2
from .interval import Interval

class BBox2D:
    def __init__(self, v0: Vector2, v1: Vector2):
        self.v0 = v0
        self.v1 = v1
    
    def __str__(self) -> str:
        return f"BBox2D({self.v0} ~ {self.v1})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def center(self) -> Vector2:
        return 0.5 * (self.v0 + self.v1)

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
    
    @classmethod
    def Union(cls, bbox0: BBox2D, bbox1: BBox2D) -> BBox2D:
        workingValues = BBox2D.WorkingValues()
        for point in [bbox0.v0, bbox0.v1, bbox1.v0, bbox1.v1]:
            workingValues.Update(point)
        return workingValues.ToBBox2D()
    
    @classmethod
    def Intersection(cls, bbox0: BBox2D, bbox1: BBox2D) -> BBox2D:
        xIntersection = Interval.Intersection(bbox0.xInterval, bbox1.xInterval)
        if xIntersection is None:
            return None
        yIntersection = Interval.Intersection(bbox0.yInterval, bbox1.yInterval)
        if yIntersection is None:
            return None
        return BBox2D(
            v0=Vector2(x=xIntersection.min, y=yIntersection.min),
            v1=Vector2(x=xIntersection.max, y=yIntersection.max)
        )
    
    @property
    def area(self) -> float:
        return self.xInterval.length * self.yInterval.length
