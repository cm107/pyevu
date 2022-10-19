from __future__ import annotations
import math
from .vector2 import Vector2
from .bbox2d import BBox2D
from typing import Callable, TypeVar, Generic, Union, overload
import numpy as np

T = TypeVar('T')

class Grid2D(Generic[T]):
    def __init__(
        self,
        width: int, height: int,
        cellWidth: float=1, cellHeight: float=1,
        origin: Vector2=Vector2.zero,
        countCorners: bool=True
    ):
        self._width = width; self._height = height
        self._cellWidth = cellWidth; self._cellHeight = cellHeight
        self._origin = origin
        self._countCorners = countCorners
        if self._countCorners:
            self._obj_arr = np.empty(shape=(width+1, height+1), dtype=np.object_)
        else:
            self._obj_arr = np.empty(shape=(width, height), dtype=np.object_)
        self.OnGridObjectChanged: list[Callable[[Vector2],]] = []

    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height
    
    @property
    def cellWidth(self) -> float:
        return self._cellWidth
    
    @property
    def cellHeight(self) -> float:
        return self._cellHeight
    
    @property
    def origin(self) -> Vector2:
        return self._origin

    @overload
    def __getitem__(self, key: tuple) -> T: ...

    @overload
    def __getitem__(self, key: Vector2) -> T: ...

    def __getitem__(self, key: Union[tuple, Vector2]) -> T:
        if type(key) is Vector2:
            x, y = tuple(key)
        elif type(key) is tuple:
            x, y = key
        else:
            raise TypeError
        return self._obj_arr.__getitem__((x, y))

    @overload
    def __setitem__(self, key: tuple, value: T): ...

    @overload
    def __setitem__(self, key: Vector2, value: T): ...

    def __setitem__(self, key: Union[tuple, Vector2], value: T):
        if type(key) is Vector2:
            x, y = tuple(key)
        elif type(key) is tuple:
            x, y = key
        else:
            raise TypeError
        self._obj_arr.__setitem__((x, y), value)
        
        if len(self.OnGridObjectChanged) > 0:
            coord = Vector2(x,y)
            for subscriber in self.OnGridObjectChanged:
                subscriber(coord)

    @property
    def gridBoundingBox(self) -> BBox2D:
        if self._countCorners:
            return BBox2D(v0=Vector2.zero, v1=Vector2(self.width, self.height))
        else:
            return BBox2D(v0=Vector2.zero, v1=Vector2(self.width-1, self.height-1))

    def World2GridCoord(self, worldCoord: Vector2, max: bool=False) -> Vector2:
        relPosition = worldCoord - self.origin
        
        def round(val: float, max: bool) -> int:
            return math.ceil(val) if max else math.floor(val)
        
        return Vector2(
            x=round(relPosition.x / self.cellWidth, max=max),
            y=round(relPosition.y / self.cellHeight, max=max)
        )

    def Grid2WorldPosition(self, gridCoord: Vector2) -> Vector2:
        return Vector2(
            x=self.origin.x + gridCoord.x * self.cellWidth,
            y=self.origin.y + gridCoord.y * self.cellHeight
        )
    
    def GetWorldPositionIfValid(self, gridCoord: Vector2) -> Vector2:
        if self.gridBoundingBox.Contains(gridCoord):
            return self.Grid2WorldPosition(gridCoord)
        else:
            return None

    def LoopCoords(self, loopFunc: Callable[[Grid2D[T], Vector2],], bbox: BBox2D=None):
        if bbox is not None:
            bbox = BBox2D.Intersection(bbox, self.gridBoundingBox)
            if bbox is None: # No intersection. BBox is completely outside of grid.
                return
        else:
            bbox = self.gridBoundingBox

        xmin = bbox.v0.x; xmax = bbox.v1.x
        ymin = bbox.v0.y; ymax = bbox.v1.y
        
        for y in range(ymin, ymax+1, 1):
            for x in range(xmin, xmax+1, 1):
                gridCoord = Vector2(x,y)
                loopFunc(self, gridCoord)
