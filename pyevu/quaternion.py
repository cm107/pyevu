from __future__ import annotations
from typing import Union
from pyquaternion import Quaternion as pyQuaternion
from .vector3 import Vector3
import math
import numpy as np

class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

        # TODO: Implement conversion between euler angles.
        # This could be complicated since unity uses a left-handed coordinate system.
    
    def __str__(self) -> str:
        return self.ToPyquaternion().__str__() # yoink
    
    def __repr__(self) -> str:
        return self.__str__()

    def __mul__(self, other: Union[Vector3, Quaternion]) -> Union[Vector3, Quaternion]:
        if type(other) is Vector3:
            rotation_matrix = self.ToPyquaternion().rotation_matrix
            other_np = np.array(other.ToList())
            return Vector3.FromList((rotation_matrix @ other_np).tolist())
        elif type(other) is Quaternion:
            rot = self.ToPyquaternion().rotation_matrix @ other.ToPyquaternion().rotation_matrix
            quat = Quaternion.FromPyquaternion(pyQuaternion._from_matrix(rot))
            return quat
        else:
            raise TypeError(f"Can't multiply {type(self).__name__} with {type(other).__name__}")
    
    def __rmul__(self, other: Union[Vector3, Quaternion]) -> Union[Vector3, Quaternion]:
        if type(other) is Quaternion:
            rot = other.ToPyquaternion().rotation_matrix @ self.ToPyquaternion().rotation_matrix
            quat = Quaternion.FromPyquaternion(pyQuaternion._from_matrix(rot))
            return quat
        else:
            return self.__mul__(other)

    def ToPyquaternion(self) -> pyQuaternion:
        return pyQuaternion(self.w, self.x, self.y, self.z)

    @classmethod
    def FromPyquaternion(cls, pyquat: pyQuaternion) -> Quaternion:
        return Quaternion(w=pyquat.w, x=pyquat.x, y=pyquat.y, z=pyquat.z)

    @classmethod
    def AngleAxis(cls, angle: float, axis: Vector3, degrees: float=True) -> Quaternion:
        if degrees:
            angle *= math.pi / 180
        return Quaternion.FromPyquaternion(pyQuaternion(axis=axis.ToList(), angle=angle))
    
    def GetEulerAngles(self, degrees: bool=True) -> Vector3:
        # http://answers.unity.com/answers/1699764/view.html
        pitch = math.atan2(2*self.x*self.w-2*self.y*self.z, 1-2*self.x**2-2*self.z**2)
        yaw = math.atan2(2*self.y*self.w-2*self.x*self.z, 1-2*self.y**2-2*self.z**2)
        roll = math.asin(2*self.x*self.y+2*self.z*self.w)
        if degrees:
            rad2deg = 180 / math.pi
            pitch *= rad2deg
            yaw *= rad2deg
            roll *= rad2deg
        return Vector3(pitch, yaw, roll) # Is this the correct order?

    # def GetEulerAngles(self, degrees: bool=True) -> Vector3:
    #     # http://answers.unity.com/answers/416428/view.html
    #     roll  = math.atan2(2*self.y*self.w - 2*self.x*self.z, 1 - 2*self.y**2 - 2*self.z**2)
    #     pitch = math.atan2(2*self.x*self.w - 2*self.y*self.z, 1 - 2*self.x**2 - 2*self.z**2)
    #     yaw   =  math.asin(2*self.x*self.y + 2*self.z*self.w)
    #     return Vector3(roll, pitch, yaw) # is this the correct order?

    @property
    def eulerAngles(self) -> Vector3:
        return self.GetEulerAngles(degrees=True)
    
    @classmethod
    def EulerVector(cls, euler: Vector3, degrees: bool=True) -> Quaternion:
        # Returns a rotation that rotates z degrees around the z axis, x degrees around the x axis, and y degrees around the y axis.
        xRot = Quaternion.AngleAxis(euler.x, axis=Vector3.right, degrees=degrees).ToPyquaternion().rotation_matrix
        yRot = Quaternion.AngleAxis(euler.y, axis=Vector3.up, degrees=degrees).ToPyquaternion().rotation_matrix
        zRot = Quaternion.AngleAxis(euler.z, axis=Vector3.forward, degrees=degrees).ToPyquaternion().rotation_matrix
        rotationMatrix = yRot @ xRot @ zRot
        quat = Quaternion.FromPyquaternion(pyQuaternion._from_matrix(rotationMatrix))
        return quat

    @classmethod
    def Euler(cls, x: float, y: float, z: float, degrees: bool=True) -> Quaternion:
        # Returns a rotation that rotates z degrees around the z axis, x degrees around the x axis, and y degrees around the y axis.
        return Quaternion.EulerVector(Vector3(x,y,z), degrees=degrees)