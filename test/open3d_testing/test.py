from __future__ import annotations
from copy import deepcopy

import numpy as np
from pyevu.util import require_dependencies
require_dependencies('open3d')
import open3d as o3d
from pyevu import Vector3, Quat, Transform

class Color:
    def __init__(self, r: float, g: float, b: float, scale: float=1.0):
        self.r = r
        self.b = b
        self.g = g
        self._scale = scale
    
    def __str__(self) -> str:
        if self._scale <= 1.0:
            return f"{type(self).__name__}({self.r},{self.g},{self.b})"
        else:
            return f"{type(self).__name__}({round(self.r)},{round(self.g)},{round(self.b)})"

    def __iter__(self) -> list[float]:
        return iter([self.r, self.g, self.b])

    def __repr__(self) -> str:
        return self.__str__()
    
    def __key(self) -> tuple:
        print(f"{list(self)=}")
        return tuple([self.__class__] + list(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented

    @property
    def scale(self) -> float:
        return self._scale
    
    @scale.setter
    def scale(self, value: float):
        ratio = value / self._scale
        self.r *= ratio; self.g *= ratio; self.b *= ratio
        self._scale = value
    
    def Rescale(self, scale: float) -> Color:
        c = deepcopy(self)
        c.scale = scale
        return c

    @classmethod
    @property
    def red(self) -> Color:
        return Color(1,0,0)

    @classmethod
    @property
    def green(self) -> Color:
        return Color(0,1,0)

    @classmethod    
    @property
    def blue(self) -> Color:
        return Color(0,0,1)

    @staticmethod
    def testbench():
        assert Color.red.Rescale(255) == Color(255, 0, 0, scale=255)
        assert list(Color.red) == [1,0,0]
        assert tuple(Color.red) == (1,0,0)
        print(f"{Color.__name__} testbench passed.")

class Artist:
    @property
    def geometries(self) -> list:
        return []

    def add_artist(self, figure: Figure):
        for g in self.geometries:
            figure.add_geometry(g)
        figure.artists.append(self)
    
    def update(self):
        pass

class Box(Artist):
    def __init__(self, transform: Transform, size: Vector3=Vector3.one, color: Color=None):
        self.transform = transform
        self.size = size
        self._box = o3d.geometry.TriangleMesh.create_box(*list(size))
        if color is not None:
            n_vertices = len(self._box.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = list(color)
            self._box.vertex_colors = o3d.utility.Vector3dVector(colors)
        self._box.compute_vertex_normals()
        self.update()
    
    @property
    def geometries(self) -> list:
        return [self._box]

    def update(self):
        print(f"{self.transform.localToWorldMatrix=}")
        mat = self.transform.localToWorldMatrix
        offset = Transform(position=self.size * -0.5*0.5, rotation=Quat.identity).localToWorldMatrix # ???
        mat = offset @ mat
        self._box.transform(mat)

class XYZMarker(Artist):
    def __init__(self, transform: Transform, size: float=1.0):
        self.transform = transform
        self.size = size
        self._marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.size)
        self.update()
    
    @property
    def geometries(self) -> list:
        return [self._marker]
    
    def update(self):
        self._marker.transform(self.transform.localToWorldMatrix)

class Figure:
    def __init__(
        self, window_name: str="Open3D", width: int=1920, height: int=1080,
        with_key_callbacks=False
    ):
        if with_key_callbacks:
            self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        else:
            self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(
            window_name=window_name, width=width, height=height)

        self.artists: list[Artist] = []
    
    def add_geometry(self, geometry):
        self.visualizer.add_geometry(geometry)

    def view_init(self, azim=-60, elev=30):
        vc = self.visualizer.get_view_control()
        pcp = vc.convert_to_pinhole_camera_parameters()
        distance = np.linalg.norm(pcp.extrinsic[:3, 3]).tolist()
        
        # azimuth and elevation are defined in world frame
        R_azim = Quat.AngleAxis(azim, axis=Vector3.up)
        R_elev = Quat.AngleAxis(elev, Vector3.right)
        camera = Transform(position=Vector3.forward * distance, rotation=R_azim * R_elev)
        pcp.extrinsic = camera.localToWorldMatrix
        vc.convert_from_pinhole_camera_parameters(pcp)

    def show(self):
        self.visualizer.run()
        self.visualizer.destroy_window()

    def save_image(self, path: str):
        self.visualizer.capture_screen_image(path, True)

    def animate(self, n_frames: int, loop: bool=False, callback=None, fargs: tuple=()):
        initialized = False
        window_open = True
        while window_open and (loop or not initialized):
            for i in range(n_frames):
                if callback is not None:
                    callback(i, *fargs)
                for artist in self.artists:
                    artist.update()

                window_open = self.visualizer.poll_events()
                if not window_open:
                    break
                self.visualizer.update_renderer()
            initialized = True

fig = Figure()
box_transform = Transform.identity
box = Box(
    transform=box_transform,
    size=Vector3.one, color=Color.red
);
box.add_artist(fig)
marker = XYZMarker(box_transform, size=1.0)
marker.add_artist(fig)

def rotate_box(i: int, *fargs):
    box_transform.localRotation *= Quat.AngleAxis(5, Vector3.up)
    box_transform.localPosition += Vector3.up * 0.1

fig.view_init(azim=0, elev=0)
fig.animate(n_frames=10000, loop=True, callback=rotate_box, fargs=()) # Not moving for some reason.
fig.show()