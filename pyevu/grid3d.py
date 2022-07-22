from __future__ import annotations
import math
from .vector3 import Vector3
from .bbox3d import BBox3D
from typing import Callable, TypeVar, Generic, Union, overload
import numpy as np

T = TypeVar('T')

class Grid3D(Generic[T]):
    def __init__(
        self,
        width: int, depth: int, height: int,
        cellWidth: float=1, cellDepth: float=1, cellHeight: float=1,
        origin: Vector3=Vector3.zero
    ):
        self._width = width; self._depth = depth; self._height = height
        self._cellWidth = cellWidth; self._cellDepth = cellDepth; self._cellHeight = cellHeight
        self._origin = origin
        self._obj_arr = np.empty(shape=(width+1, depth+1, height+1), dtype=np.object_)
        self.OnGridObjectChanged: list[Callable[[Vector3],]] = []

    @property
    def width(self) -> int:
        return self._width
    
    @property
    def depth(self) -> int:
        return self._depth
    
    @property
    def height(self) -> int:
        return self._height
    
    @property
    def cellWidth(self) -> float:
        return self._cellWidth
    
    @property
    def cellDepth(self) -> float:
        return self._cellDepth
    
    @property
    def cellHeight(self) -> float:
        return self._cellHeight
    
    @property
    def origin(self) -> Vector3:
        return self._origin

    @overload
    def __getitem__(self, key: tuple) -> T: ...

    @overload
    def __getitem__(self, key: Vector3) -> T: ...

    def __getitem__(self, key: Union[tuple, Vector3]) -> T:
        if type(key) is Vector3:
            x, y, z = key.ToTuple()
        elif type(key) is tuple:
            x, y, z = key
        else:
            raise TypeError
        return self._obj_arr.__getitem__((x, z, y))

    @overload
    def __setitem__(self, key: tuple, value: T): ...

    @overload
    def __setitem__(self, key: Vector3, value: T): ...

    def __setitem__(self, key: Union[tuple, Vector3], value: T):
        if type(key) is Vector3:
            x, y, z = key.ToTuple()
        elif type(key) is tuple:
            x, y, z = key
        else:
            raise TypeError
        self._obj_arr.__setitem__((x, z, y), value)
        
        if len(self.OnGridObjectChanged) > 0:
            coord = Vector3(x,y,z)
            for subscriber in self.OnGridObjectChanged:
                subscriber(coord)

    @property
    def gridBoundingBox(self) -> BBox3D:
        return BBox3D(v0=Vector3.zero, v1=Vector3(self.width, self.height, self.depth))

    def World2GridCoord(self, worldCoord: Vector3, max: bool=False) -> Vector3:
        relPosition = worldCoord - self.origin
        
        def round(val: float, max: bool) -> int:
            return math.ceil(val) if max else math.floor(val)
        
        return Vector3(
            x=round(relPosition.x / self.cellWidth, max=max),
            y=round(relPosition.y / self.cellHeight, max=max),
            z=round(relPosition.z / self.cellDepth, max=max)
        )

    def Grid2WorldCoord(self, gridCoord: Vector3) -> Vector3:
        return Vector3(
            x=self.origin.x + gridCoord.x * self.cellWidth,
            y=self.origin.y + gridCoord.y * self.cellHeight,
            z=self.origin.z + gridCoord.z * self.cellDepth
        )
    
    def GetWorldPositionIfValid(self, gridCoord: Vector3) -> Vector3:
        if self.gridBoundingBox.Contains(gridCoord):
            return self.Grid2WorldCoord(gridCoord)
        else:
            return None

    def LoopCoords(self, loopFunc: Callable[[Grid3D[T], Vector3],], bbox: BBox3D=None):
        if bbox is not None:
            bbox = BBox3D.Intersection(bbox, self.gridBoundingBox)
            if bbox is None: # No intersection. BBox is completely outside of grid.
                return

        if bbox is not None:
            xmin = bbox.v0.x; xmax = bbox.v1.x
            ymin = bbox.v0.y; ymax = bbox.v1.y
            zmin = bbox.v0.z; zmax = bbox.v1.z
        else:
            xmin = 0; xmax = self.width
            ymin = 0; ymax = self.height
            zmin = 0; zmax = self.depth
        
        for y in range(ymin, ymax+1, 1):
            for z in range(zmin, zmax+1, 1):
                for x in range(xmin, xmax+1, 1):
                    gridCoord = Vector3(x,y,z)
                    loopFunc(self, gridCoord)

    @staticmethod
    def debug0():
        class TestGridObject:
            def __init__(self, a: float, b: float):
                self.a = a
                self.b = b
            
            def __str__(self) -> str:
                return f"GridObject(a={self.a},b={self.b})"
            
            def __repr__(self) -> str:
                return self.__str__()

        def test_callback(coord: Vector3):
            print(f"Value of grid at coordinate {coord} changed.")

        grid = Grid3D[TestGridObject](11,14,20, 0.3,0.5,1.6)
        grid.OnGridObjectChanged.append(test_callback)
        grid[0,0,0] = TestGridObject(2, 4)
        print(f"{grid._obj_arr[0,0,0]=}")
        print(f"{grid._obj_arr.dtype=}")
        print(f"{grid[0,0,0]=}")
        print(f"{grid[0,0,0].a=}")
        print(f"{grid.World2GridCoord(Vector3(2.4, 7.8, 5.3))=}")
    
    @staticmethod
    def debug():
        class SDFObject:
            def __init__(self, signedDistance: float):
                self.signedDistance = signedDistance
            
        grid = Grid3D[SDFObject](10,10,10, 1,1,1, origin=Vector3.one * -5)

        class Sphere:
            def __init__(self, center: Vector3, radius: float):
                self.center = center
                self.radius = radius
            
            def SDF(self, p: Vector3) -> float:
                return Vector3.Distance(p, self.center) - self.radius
            
            @staticmethod
            def SDF_merge(p: Vector3, spheres: list[Sphere]) -> float:
                return min(*[sphere.SDF(p) for sphere in spheres])
            
            @staticmethod
            def SDF_intersection(p: Vector3, spheres: list[Sphere]) -> float:
                return max(*[sphere.SDF(p) for sphere in spheres])
        
        sphere0 = Sphere(center=Vector3.zero, radius=3)
        sphere1 = Sphere(center=Vector3(1.5, 0, 1.5), radius=2)
        spheres = [sphere0, sphere1]

        def initGrid(g: Grid3D[SDFObject], coord: Vector3):
            p = grid.Grid2WorldCoord(coord)
            g[coord] = SDFObject(Sphere.SDF_intersection(p, spheres))
        
        def callback(coord: Vector3):
            print(f"{tuple(coord.ToList())} -> {grid[coord].signedDistance}")
        
        grid.OnGridObjectChanged.append(callback)

        grid.LoopCoords(initGrid)
