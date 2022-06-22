from __future__ import annotations
from typing import cast, TYPE_CHECKING
import numpy as np
from .vector3 import Vector3
from .quat import Quat

if TYPE_CHECKING:
    # Refer to https://www.stefaanlippens.net/circular-imports-type-hints-python.html
    from .gameObject import GameObject

class Transform:
    def __init__(self, position: Vector3, rotation: Quat, parent: Transform=None, gameObject: GameObject=None):
        self._parent = parent
        self.gameObject = gameObject
        self._worldTransformationMatrix: np.ndarray = None

        if self._parent is None:
            self._localPosition = position # world position
            self._localRotation = rotation # world rotation
        else:
            raise NotImplementedError

    #region Dunder Methods
    def __str__(self) -> str:
        return f"Transform({self.position},{self.rotation.eulerAngles})"

    def __repr__(self) -> str:
        return self.__str__()

    def __mul__(self, other: Transform) -> Transform:
        if type(other) is Transform:
            return Transform.FromTransformationMatrix(self.worldTransformationMatrix @ other.worldTransformationMatrix)
        else:
            raise TypeError(f"Can't multiply {type(self).__name__} with {type(other).__name__}")

    def __pow__(self, other: int) -> Transform:
        if type(other) is int:
            if other == 0:
                return Transform.identity
            elif other > 0:
                mat = self.worldTransformationMatrix
                result = np.eye(4)
                for i in range(other):
                    result = mat @ result
                return Transform.FromTransformationMatrix(result)
            else:
                mat = np.linalg.inv(self.worldTransformationMatrix)
                result = np.eye(4)
                for i in range(-other):
                    result = mat @ result
                return Transform.FromTransformationMatrix(result)
        else:
            raise TypeError(f"Can't raise type {type(self).__name__} to the power of a value of type {type(other).__name__}")
    #endregion

    #region Serialization
    def ToDict(self) -> dict:
        return {
            'position': self.position.ToList(),
            'rotation': self.rotation.ToList()
        } # Note: Ignoring parent on purpose.
    
    @classmethod
    def FromDict(cls, itemDict: dict) -> Transform:
        return Transform(
            position=Vector3.FromList(itemDict['position']),
            rotation=Quat.FromList(itemDict['rotation'])
        ) # Note: Ignoring parent on purpose.

    def Copy(self) -> Transform:
        return Transform.FromDict(self.ToDict())
    #endregion

    #region Core Logic
    @classmethod
    @property
    def identity(self) -> Transform:
        return Transform(position=Vector3.zero, rotation=Quat.identity)

    @property
    def parent(self) -> Transform:
        return self._parent
    
    @parent.setter
    def parent(self, transform: Transform):
        # Note: If the gameObject reference on all transforms are None, this just becomes
        #       self._parent = transform
        #       You shouldn't need to assign a reference to gameObject if you don't care about the GameObject hierarchy.
        #       The implications of not having a GameObject hierarchy, though, is that you can't move the children under a transform in tandem.
        #       With the parent member, we can know the parent of this transform, but we have no knowledge of any children without the gameObject reference.
        #       Why implement the Transform class like this? Because the implementation of the local transform only depends
        #       on this transform's transformation matrix and the transformation matrix of its parent.
        #       We do not need the transformation matrix of any of this transform's children for simple transformation matrix logic, so that is why they are not in here.
        #       Even if we did define the children logic in here as well, it would just make the code look messier and the class itself less flexible.
        #

        if self._parent is not None and self._parent.gameObject is not None:
            # Unregister this GameObject from current parent's children.
            self._parent.gameObject.children.remove(self.gameObject)
        
        self._parent = transform

        if self.gameObject is not None and self._parent is not None:
            # Register this GameObject to new parent's children.
            self._parent.gameObject.children.append(self.gameObject)
            # print(f"Added {self.gameObject.name} to children of {self._parent.gameObject.name}")

    @property
    def position(self) -> Vector3:
        return Vector3.FromNumpy(self.worldTransformationMatrix[:3, 3])

    def SetPosition(self, position: Vector3, delay_update: bool=False):
        # worldPoint = t_world @ X
        # localPoint = worldToLocalMatrix @ worldPoint
        # newLocalPoint = worldToLocalMatrix @ newWorldPoint
        self._localPosition = self.worldToLocalMatrix @ position
        if not delay_update:
            self.UpdateWorldTransformationMatrix()

    @position.setter
    def position(self, position: Vector3):
        self.SetPosition(position, delay_update=False)

    @property
    def rotation(self) -> Quat:
        return Quat.from_rotation_matrix(self.worldTransformationMatrix[:3, :3])

    def SetRotation(self, rotation: Quat, delay_update: bool=False):
        # Assuming that a change in world rotation is the same as a change in local rotation
        # diff * world_q1 = world_q2
        # => diff = world_q2 * world_q1.inverse
        # diff * local_q1 = local_q2
        diffRot = rotation * self.rotation.inverse
        self._localRotation = diffRot * self.localRotation
        if not delay_update:
            self.UpdateWorldTransformationMatrix()

    @rotation.setter
    def rotation(self, rotation: Quat):
        self.SetRotation(rotation, delay_update=False)

    def SetPositionAndRotation(self, position: Vector3, rotation: Quat):
        """
        Sets the world space position and rotation.
        Setting position and rotation individually updates the world transformation matrix twice,
        but using this method updates the world transformation matrix just once, and is therefore
        more efficient.
        """
        self.SetPosition(position, delay_update=True)
        self.SetRotation(rotation, delay_update=True)
        self.UpdateWorldTransformationMatrix()

    @property
    def localPosition(self) -> Vector3:
        return self._localPosition
    
    @localPosition.setter
    def localPosition(self, position: Vector3):
        self._localPosition = position
        self.UpdateWorldTransformationMatrix()

    @property
    def localRotation(self) -> Quat:
        return self._localRotation

    @localRotation.setter
    def localRotation(self, rotation: Quat):
        self._localRotation = rotation
        self.UpdateWorldTransformationMatrix()

    def SetLocalPositionAndRotation(self, position: Vector3, rotation: Quat):
        self._localPosition = position
        self._localRotation = rotation
        self.UpdateWorldTransformationMatrix()

    @property
    def localTransformationMatrix(self) -> np.ndarray:
        mat = self.localRotation.rotation_matrix
        mat = np.pad(mat, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        mat[0:4, 3] = np.array(self.localPosition.ToList() + [1])
        return mat

    def UpdateWorldTransformationMatrix(self):
        if self.parent is not None:
            self._worldTransformationMatrix = self.parent.worldTransformationMatrix @ self.localTransformationMatrix
        else:
            self._worldTransformationMatrix = self.localTransformationMatrix
        if self.gameObject is not None:
            self.gameObject._ChildrenUpdateWorldTransformationMatrix()

    @property
    def worldTransformationMatrix(self) -> np.ndarray:
        if self._worldTransformationMatrix is None:
            self.UpdateWorldTransformationMatrix()
        return self._worldTransformationMatrix

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
            rotation=Quat.from_rotation_matrix(transform_matrix[:3, :3]),
            parent=None,
            gameObject=None
        )

    @property
    def localToWorldMatrix(self) -> np.ndarray:
        """Matrix that transforms a point from local space into world space"""
        # localPoint = t_local @ X
        # worldPoint = t_world @ X
        # worldPoint = mat @ localPoint
        # => (t_world @ X) = mat @ (t_local @ X)
        # => (t_world) @ X = (mat @ t_local) @ X
        # => t_world = mat @ t_local
        # => mat = t_world @ t_local.inv
        return self.worldTransformationMatrix @ np.linalg.inv(self.localTransformationMatrix)
    
    @property
    def worldToLocalMatrix(self) -> np.ndarray:
        """Matrix that transforms a point from world space into local space"""
        # localPoint = t_local @ X
        # worldPoint = t_world @ X
        # localPoint = mat @ worldPoint
        # => (t_local @ X) = mat @ (t_world @ X)
        # => (t_local) @ X = (mat @ t_world) @ X
        # => t_local = mat @ t_world
        # => mat = t_local @ t_world.inv
        return self.localTransformationMatrix @ np.linalg.inv(self.worldTransformationMatrix)
    #endregion

    #region Hierarchy Utility
    @property
    def path(self) -> str:
        if self.gameObject is None:
            return ""
        else:
            if self.parent is None:
                return self.gameObject.name
            else:
                return f"{self.parent.path}/{self.gameObject.name}"
    
    def Find(self, path: str) -> Transform:
        if self.gameObject is None:
            return None
        else:
            if path == self.path:
                return self
            for child in self.gameObject.children:
                if child.transform.path == path:
                    return child.transform
                else:
                    match = child.transform.Find(path)
                    if match is not None:
                        return match
            return None

    def FindChild(self, name: str) -> Transform:
        if self.gameObject is None:
            return None
        else:
            for child in self.gameObject.children:
                if child.name == name:
                    return child.transform
            return None
    
    @property
    def hierarchyPaths(self) -> list[str]:
        paths: list[str] = [self.path]
        
        if self.gameObject is None:
            return paths
        else:
            for child in self.gameObject.children:
                paths.extend(child.transform.hierarchyPaths)
            return paths

    def PrintHierarchy(self):
        print("Hierarchy:")
        for path in self.hierarchyPaths:
            t = self.Find(path)
            print(f"\t{path}: {t}")
    #endregion

    #region Directions
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
    #endregion

    #region Applying To Vertices
    def TransformPoint(self, point: Vector3) -> Vector3:
        return Vector3.FromNumpy((self.worldTransformationMatrix @ point.mat4.T).T.reshape(-1)[:3])

    def TransformPoints(self, points: list[Vector3]) -> list[Vector3]:
        return [self.TransformPoint(point) for point in points]
    #endregion
