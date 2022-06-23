from turtle import update
import numpy as np
from matplotlib import artist
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, axes3d
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable
from .. import Transform, Quat, Vector3


class XYZMarker(artist.Artist):
    """A Matplotlib artist that displays a frame represented by its basis.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    label : str, optional (default: None)
        Name of the frame

    s : float, optional (default: 1)
        Length of basis vectors

    draw_label_indicator : bool, optional (default: True)
        Controls whether the line from the frame origin to frame label is
        drawn.

    Other arguments except 'c' and 'color' are passed on to Line3D.
    """
    def __init__(self, A2B, label=None, s=1.0, **kwargs):
        super(XYZMarker, self).__init__()

        if "c" in kwargs:
            kwargs.pop("c")
        if "color" in kwargs:
            kwargs.pop("color")

        self.draw_label_indicator = kwargs.pop("draw_label_indicator", True)

        self.s = s

        self.x_axis = Line3D([], [], [], color="r", **kwargs)
        self.y_axis = Line3D([], [], [], color="g", **kwargs)
        self.z_axis = Line3D([], [], [], color="b", **kwargs)

        self.draw_label = label is not None
        self.label = label

        if self.draw_label:
            if self.draw_label_indicator:
                self.label_indicator = Line3D([], [], [], color="k", **kwargs)
            self.label_text = Text3D(0, 0, 0, text="", zdir="x")

        self.set_data(A2B, label)

    def set_data(self, A2B, label=None):
        """Set the transformation data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        label : str, optional (default: None)
            Name of the frame
        """
        R = A2B[:3, :3]
        p = A2B[:3, 3]

        for d, b in enumerate([self.x_axis, self.y_axis, self.z_axis]):
            b.set_data(np.array([p[0], p[0] + self.s * R[0, d]]),
                       np.array([p[1], p[1] + self.s * R[1, d]]))
            b.set_3d_properties(np.array([p[2], p[2] + self.s * R[2, d]]))

        if self.draw_label:
            if label is None:
                label = self.label
            label_pos = p + 0.5 * self.s * (R[:, 0] + R[:, 1] + R[:, 2])

            if self.draw_label_indicator:
                self.label_indicator.set_data(
                    np.array([p[0], label_pos[0]]),
                    np.array([p[1], label_pos[1]]))
                self.label_indicator.set_3d_properties(
                    np.array([p[2], label_pos[2]]))

            self.label_text.set_text(label)
            self.label_text.set_position([label_pos[0], label_pos[1]])
            self.label_text.set_3d_properties(label_pos[2], zdir="x")

    @artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        """Draw the artist."""
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            b.draw(renderer, *args, **kwargs)
        if self.draw_label:
            if self.draw_label_indicator:
                self.label_indicator.draw(renderer, *args, **kwargs)
            self.label_text.draw(renderer, *args, **kwargs)
        super(XYZMarker, self).draw(renderer, *args, **kwargs)

    def add_frame(self, axis):
        """Add the frame to a 3D axis."""
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            axis.add_line(b)
        if self.draw_label:
            if self.draw_label_indicator:
                axis.add_line(self.label_indicator)
            axis._add_text(self.label_text)

class XYZObject(XYZMarker):
    def __init__(self, update: Callable[[int, int], Transform], label: str=None, s: float=1.0, **kwargs):
        self._update: Callable[[int, int], Transform] = update
        A2B = self._update(0, 100).worldTransformationMatrix
        super().__init__(A2B=A2B, label=label, s=s, **kwargs)

class XYZObjectScene:
    def __init__(
        self, obj_list: list[XYZObject],
        n_frames: int=50,
        xlim: tuple[int, int]=(-1, 1),
        ylim: tuple[int, int]=(-1, 1),
        zlim: tuple[int, int]=(-1, 1),
    ):

        self.fig = plt.figure(figsize=(5, 5))

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.invert_zaxis()
        self.ax.view_init(vertical_axis='y')

        self.n_frames = n_frames

        self.obj_list = obj_list
        for obj in self.obj_list:
            obj.add_frame(self.ax)
    
    def update_frame(self, step: int, n_frames: int) -> list[XYZObject]:
        for obj in self.obj_list:
            t = obj._update(step, n_frames)
            obj.set_data(t.worldTransformationMatrix)
        return self.obj_list
    
    def run(self):
        anim = animation.FuncAnimation(
            self.fig, self.update_frame, self.n_frames, fargs=(self.n_frames,), interval=50,
            blit=False)
        plt.show()

from .. import GameObject
import sys

class GameObjectXYZMarker(XYZObject):
    def __init__(self, gameObject: GameObject, s: float=1.0, **kwargs):
        self.gameObject = gameObject
        super().__init__(
            update=self.get_transform_func(),
            label=self.gameObject.name,
            s=s,
            **kwargs
        )
    
    def get_transform_func(self):
        def _func(*args, **kwargs) -> Transform:
            return self.gameObject.transform
        return _func

class SimpleLine(artist.Artist):
    def __init__(self, start: Vector3, end: Vector3, **kwargs):
        super().__init__()
        if 'c' not in kwargs and 'color' not in kwargs:
            kwargs['color'] = 'k'
        self.line = Line3D(xs=[], ys=[], zs=[], **kwargs)
        self.set_data(start=start, end=end)
    
    def set_data(self, start: Vector3, end: Vector3):
        self.line.set_data_3d(
            [start.x, end.x],
            [start.y, end.y],
            [start.z, end.z]
        )
        # self.line.set_3d_properties(zs=start.z)

    @artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        """Draw the artist."""
        self.line.draw(renderer, *args, **kwargs)
        super().draw(renderer, *args, **kwargs)
    
    def add_frame(self, axis):
        """Add the frame to a 3D axis."""
        axis.add_line(self.line)

class GameObjectMarkerScene:
    def __init__(
        self, obj_dict: dict[str, GameObjectXYZMarker],
        n_frames: int=50,
        xlim: tuple[int, int]=(-1, 1),
        ylim: tuple[int, int]=(-1, 1),
        zlim: tuple[int, int]=(-1, 1),
        update_transforms: Callable[[dict[str, GameObject], list[str], int, int], None]=None,
        draw_lines_to_children: bool=False
    ):
        self.fig = plt.figure(figsize=(5, 5))

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.invert_zaxis()
        self.ax.view_init(vertical_axis='y')

        self._pressed_keys: list[str] = []
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_release)

        self.n_frames = n_frames

        self.obj_dict = obj_dict
        for obj in self.obj_dict.values():
            obj.add_frame(self.ax)
        
        self.line_map: dict[str, SimpleLine] = {}
        self.draw_lines_to_children = draw_lines_to_children
        if self.draw_lines_to_children:
            for go in [obj.gameObject for obj in self.obj_dict.values()]:
                for child in go.children:
                    line_name = f"{go.name}-{child.name}"
                    start = go.transform.position
                    end = child.transform.position
                    line = SimpleLine(start=start, end=end, color='blue')
                    self.line_map[line_name] = line
                    line.add_frame(self.ax)
        self._update_transforms = update_transforms
    
    def on_press(self, event):
        key = event.key
        sys.stdout.flush()
        if key not in self._pressed_keys:
            self._pressed_keys.append(key)
    
    def on_release(self, event):
        key = event.key
        sys.stdout.flush()
        if key in self._pressed_keys:
            self._pressed_keys.remove(key)

    def update_frame(self, step: int, n_frames: int) -> list[GameObjectXYZMarker]:
        if self._update_transforms is not None:
            # print(f"{self._pressed_keys=}")
            self._update_transforms(
                {key: marker.gameObject for key, marker in self.obj_dict.items()},
                self._pressed_keys, step, n_frames
            )

        for obj in self.obj_dict.values():
            t = obj._update(step, n_frames)
            obj.set_data(t.worldTransformationMatrix)
        if self.draw_lines_to_children:
            for go in [obj.gameObject for obj in self.obj_dict.values()]:
                for child in go.children:
                    line_name = f"{go.name}-{child.name}"
                    start = go.transform.position
                    end = child.transform.position
                    self.line_map[line_name].set_data(start=start, end=end)
        return self.obj_dict.values()
    
    def run(self):
        anim = animation.FuncAnimation(
            self.fig, self.update_frame, self.n_frames, fargs=(self.n_frames,), interval=50,
            blit=False)
        plt.show()