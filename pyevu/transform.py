from __future__ import annotations
from typing import cast, List
import numpy as np
from .vector3 import Vector3
from .quaternion import Quaternion

class Transform:
    def __init__(self, position: Vector3, rotation: Quaternion, parent: Transform=None):
        self.position = position # world position
        self.rotation = rotation # world rotation
        self.parent = parent

    def __str__(self) -> str:
        return f"Transform({self.position},{self.rotation})"

    def __repr__(self) -> str:
        return self.__str__()

    def __mul__(self, other: Transform) -> Transform:
        if type(other) is Transform:
            return Transform.FromTransformationMatrix(self.transformationMatrix @ other.transformationMatrix)
        else:
            raise TypeError(f"Can't multiply {type(self).__name__} with {type(other).__name__}")

    def __pow__(self, other: int) -> Transform:
        if type(other) is int:
            if other == 0:
                return Transform.identity
            elif other > 0:
                mat = self.transformationMatrix
                result = Transform.identity.transformationMatrix
                for i in range(other):
                    result = mat @ result
                return Transform.FromTransformationMatrix(result)
            else:
                mat = self.inverse.transformationMatrix
                result = Transform.identity.transformationMatrix
                for i in range(-other):
                    result = mat @ result
                return Transform.FromTransformationMatrix(result)
        else:
            raise TypeError(f"Can't raise type {type(self).__name__} to the power of a value of type {type(other).__name__}")

    def ToDict(self) -> dict:
        return {
            'position': self.position.ToList(),
            'rotation': self.rotation.ToList()
        } # Note: Ignoring parent on purpose.
    
    @classmethod
    def FromDict(cls, itemDict: dict) -> Transform:
        return Transform(
            position=Vector3.FromList(itemDict['position']),
            rotation=Quaternion.FromList(itemDict['rotation'])
        ) # Note: Ignoring parent on purpose.

    def Copy(self) -> Transform:
        return Transform.FromDict(self.ToDict())

    @classmethod
    @property
    def identity(self) -> Transform:
        return Transform(position=Vector3.zero, rotation=Quaternion.identity)

    @property
    def localTransform(self) -> Transform:
        if self.parent is None:
            return self.Copy()
        else:
            # thisGlobalTransform = thisLocalTransform * parentGlobalTransform
            # => thisGlobalTransform * parentGlobalTransform.inverse = thisLocalTransform * parentGlobalTransform * parentGlobalTransform.inverse
            # => thisGlobalTransform * parentGlobalTransform.inverse = thisLocalTransform
            return self * self.parent.inverse
    
    @localTransform.setter
    def localTransform(self, transform: Transform):
        # Note: Need to be careful here not to copy any references and also to preserve the parent reference.
        if self.parent is None:
            self.position = transform.position.Copy()
            self.rotation = transform.rotation.Copy()
        else:
            # thisGlobalTransform = thisLocalTransform * parentGlobalTransform
            result = transform * self.parent
            self.position = result.position.Copy()
            self.rotation = result.rotation.Copy()

    @property
    def localPosition(self) -> Vector3:
        return self.localTransform.position
    
    @localPosition.setter
    def localPosition(self, position: Vector3):
        local = self.localTransform
        local.position = position
        self.localTransform = local

    @property
    def localRotation(self) -> Quaternion:
        return self.localTransform.rotation

    @localRotation.setter
    def localRotation(self, rotation: Quaternion):
        local = self.localTransform
        local.rotation = rotation
        self.localTransform = local

    @property
    def transformationMatrix(self) -> np.ndarray:
        mat = self.rotation.rotation_matrix
        mat = np.pad(mat, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        mat[0:4, 3] = np.array(self.position.ToList() + [1])
        return mat

    @classmethod
    def FromTransformationMatrix(cls, transform_matrix: np.ndarray) -> Transform:
        expectedShape = (4, 4)
        if transform_matrix.shape != expectedShape:
            raise ValueError(f"Expected transform_matrix to be of shape {expectedShape}. Encountered {transform_matrix.shape}")
        expectedBottomRow = np.array([0,0,0,1])
        if not all((transform_matrix[-1] == expectedBottomRow).tolist()):
            raise ValueError(f"Invalid transformation_matrix. Expected bottom row to be {expectedBottomRow}. Encountered {transform_matrix[-1]}")
        
        return Transform(
            position=Vector3.FromNumpy(transform_matrix[:3, 3]),
            rotation=Quaternion.FromRotationMatrix(transform_matrix[:3, :3])
        )

    @property
    def inverse(self) -> Transform:
        return Transform.FromTransformationMatrix(np.linalg.inv(self.transformationMatrix))

    def TransformPoint(self, point: Vector3) -> Vector3:
        return Vector3.FromNumpy((self.transformationMatrix @ point.mat4.T).T.reshape(-1)[:3])

    def TransformPoints(self, points: List[Vector3]) -> List[Vector3]:
        return [self.TransformPoint(point) for point in points]

    @property
    def forward(self) -> Vector3:
        return self.rotation * Vector3.forward

    @property
    def backward(self) -> Vector3:
        return self.rotation * Vector3.backward

    @property
    def left(self) -> Vector3:
        return self.rotation * Vector3.left

    @property
    def right(self) -> Vector3:
        return self.rotation * Vector3.right

    @property
    def up(self) -> Vector3:
        return self.rotation * Vector3.up

    @property
    def down(self) -> Vector3:
        return self.rotation * Vector3.down
    
    @classmethod
    def Debug(cls):
        import importlib
        found = importlib.util.find_spec('plotly') is not None
        if not found:
            raise ImportError(f"Need to install plotly before debugging this class.")

        import plotly.graph_objects as go
        from typing import Tuple

        def _TransformMarker(transform: Transform, opacity: float=1.0) -> List[go.Scatter3d]:
            endpoints = [transform.position + direction for direction in [transform.right, transform.up, transform.forward]]
            startpoint = transform.position
            colors = [(255,0,0), (0,255,0), (0,0,255)]
            names = ['x', 'y', 'z']

            traces: List[go.Scatter3d] = []

            for endpoint, color, name in zip(endpoints, colors, names):
                x_lines = []
                y_lines = []
                z_lines = []
                for v in [startpoint, endpoint]:
                    x_lines.append(v.x)
                    y_lines.append(v.y)
                    z_lines.append(v.z)
                trace = go.Scatter3d(
                    x=x_lines,
                    y=y_lines,
                    z=z_lines,
                    mode='lines',
                    # name=name,
                    line=dict(
                        color=f"rgba({color[0]},{color[1]},{color[2]},{opacity})",
                        width=2
                    )
                )
                traces.append(trace)
            
            return traces

        def _PlotGroups(
            groups: List[go.Scatter3d],
            cameraUp: Vector3=Vector3.up,
            cameraCenter: Vector3=Vector3.zero,
            cameraEye: Vector3=Vector3.forward,
            xrange: Tuple[float, float]=[-10, 10],
            yrange: Tuple[float, float]=[-10, 10],
            zrange: Tuple[float, float]=[-10, 10],
            xticks: int=4, yticks: int=4, zticks: int=4
        ):
            fig = go.Figure()
            for group in groups:
                fig.add_trace(group)
            
            fig.update_layout(
                scene = dict(
                    xaxis = dict(nticks=xticks, range=xrange,),
                    yaxis = dict(nticks=yticks, range=yrange,),
                    zaxis = dict(nticks=zticks, range=zrange,),
                    aspectratio=dict(x=1,y=1,z=1)
                ),
                width=700,
                margin=dict(r=20, l=10, b=10, t=10),
            )

            name = 'default'
            # Default parameters which are used when `layout.scene.camera` is not provided
            camera = dict(
                up=dict(x=cameraUp.x, y=cameraUp.y, z=cameraUp.z),
                center=dict(x=cameraCenter.x, y=cameraCenter.y, z=cameraCenter.z),
                eye=dict(x=cameraEye.x, y=cameraEye.y, z=cameraEye.z)
            )

            fig.update_layout(scene_camera=camera, title=name)
            fig.update_layout(showlegend=False)
            fig.show()
        
        order = 'zxy'
        transformList: List[Transform] = [Transform.identity]
        for i in range(9):
            transformList.append(
                Transform(position=Vector3.one, rotation=Quaternion.Euler(30, 45, 60, order=order))
                * transformList[-1]
            )
        
        groups = []
        for i, transform in enumerate(transformList):
            # groups.extend(_TransformMarker(transform, opacity=1.0-0.1*i))
            groups.extend(_TransformMarker(transform, opacity=1.0))

        _PlotGroups(groups)