from __future__ import annotations
from typing import Union
import math
import numpy as np
from .vector3 import Vector3

class Quat:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    #region print string related
    @property
    def constructor_form_str(self) -> str:
        return f"Quat({self.w},{self.x},{self.y},{self.z})"

    @property
    def vector_form_str(self) -> str:
        return f"{self.w}+{self.x}i+{self.y}j+{self.z}k"

    def __str__(self) -> str:
        return self.vector_form_str
    
    def __repr__(self) -> str:
        return self.__str__()
    #endregion
    
    #region equality related
    def __key(self) -> tuple:
        return tuple([self.__class__] + list(self.__dict__.values()))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented
    #endregion

    #region operators
    def __neg__(self) -> Quat:
        return Quat(-self.w, -self.x, -self.y, -self.z)

    def __add__(self, other) -> Quat:
        if type(other) is Quat:
            return Quat(
                w=self.w+other.w,
                x=self.x+other.x,
                y=self.y+other.y,
                z=self.z+other.z
            )
        else:
            raise TypeError
    
    def __radd__(self, other) -> Quat:
        if type(other) is Quat:
            return other.__add__(self)
        else:
            raise TypeError
    
    def __sub__(self, other) -> Quat:
        if type(other) is Quat:
            return self + (-other)
        else:
            raise TypeError
    
    def __rsub__(self, other) -> Quat:
        if type(other) is Quat:
            return other + (-self)
        else:
            raise TypeError
    
    def __mul__(self, other) -> Quat:
        if type(other) is Quat:
            return Quat.hamilton_product(q0=self, q1=other)
        elif type(other) in [float, int]:
            return Quat(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError(f"Cannot multiple {type(self).__name__} with {type(other).__name__}")
    
    def __rmul__(self, other) -> Quat:
        if type(other) is Quat:
            return other.__mul__(self)
        elif type(other) in [float, int]:
            return self.__mul__(other)
        else:
            raise TypeError(f"Cannot multiple {type(other).__name__} with {type(self).__name__}")
    
    def __truediv__(self, other) -> Quat:
        if type(other) is Quat:
            return self.__mul__(other.inverse)
        elif type(other) in [float, int]:
            return self.__mul__(1/other)
        else:
            raise TypeError
    
    def __rtruediv__(self, other) -> Quat:
        if type(other) is Quat:
            return other.__mul__(self.inverse)
        else:
            raise TypeError
    #endregion

    #region array conversion
    def ToList(self) -> list[float]:
        return [self.w, self.x, self.y, self.z]
    
    @staticmethod
    def FromList(vals: list) -> Quat:
        return Quat(*vals)

    def ToNumpy(self) -> np.ndarray:
        return np.array(self.ToList())
    
    @staticmethod
    def FromNumpy(vals: np.ndarray) -> Quat:
        return Quat.FromList(vals.tolist())
    #endregion

    #region basic properties
    @property
    def real_part(self) -> float:
        return self.w
    
    @staticmethod
    def from_real_part(r: float) -> Quat:
        return Quat(r, 0, 0, 0)

    @property
    def vector_part(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    @staticmethod
    def from_vector_part(v: Vector3) -> Quat:
        return Quat(0, v.x, v.y, v.z)

    @classmethod
    @property
    def one(cls) -> Quat:
        return Quat(1, 1, 1, 1)
    
    @classmethod
    @property
    def zero(cls) -> Quat:
        return Quat(0, 0, 0, 0)
    
    @classmethod
    @property
    def identity(cls) -> Quat:
        return Quat.one

    @property
    def norm(self) -> float:
        return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
    @property
    def magnitude(self) -> float:
        return self.norm**0.5
    
    @property
    def unit(self) -> Quat:
        mag = self.magnitude
        if mag == 0:
            return None
        return self / mag
    
    @property
    def normalized(self) -> Quat:
        mag = self.magnitude
        if mag == 0:
            return Quat.zero
        return self / mag
    
    @property
    def conjugate(self) -> Quat:
        return Quat(self.w, -self.x, -self.y, -self.z)
    
    @property
    def inverse(self) -> Quat:
        norm = self.norm
        if norm == 0:
            return Quat.zero
        return self.conjugate / norm
    #endregion

    #region core logic
    @staticmethod
    def hamilton_product(q0: Quat, q1: Quat) -> Quat:
        # https://en.wikipedia.org/wiki/Quaternion
        a1a2 = q0.w * q1.w
        a1b2 = q0.w * q1.x
        a1c2 = q0.w * q1.y
        a1d2 = q0.w * q1.z

        b1a2 = q0.x * q1.w
        b1b2 = q0.x * q1.x
        b1c2 = q0.x * q1.y
        b1d2 = q0.x * q1.z

        c1a2 = q0.y * q1.w
        c1b2 = q0.y * q1.x
        c1c2 = q0.y * q1.y
        c1d2 = q0.y * q1.z

        d1a2 = q0.z * q1.w
        d1b2 = q0.z * q1.x
        d1c2 = q0.z * q1.y
        d1d2 = q0.z * q1.z

        w = a1a2 - b1b2 - c1c2 - d1d2
        x = a1b2 + b1a2 + c1d2 - d1c2
        y = a1c2 - b1d2 + c1a2 + d1b2
        z = a1d2 + b1c2 - c1b2 + d1a2

        return Quat(w, x, y, z)

    @staticmethod
    def rotate(v: Vector3, q: Union[Quat, list[Quat], tuple[Quat]]) -> Vector3:
        if type(v) is not Vector3:
            raise TypeError
        if type(q) is Quat:
            q0 = q
        elif type(q) in [list, tuple]:
            q0 = Quat.identity
            for part in list(q)[::-1]:
                if type(part) is not Quat:
                    raise TypeError
                q0 *= part
        else:
            raise TypeError

        q0 = q0.unit
        if q0 is None:
            return v
        else:
            return (q0 * Quat.from_vector_part(v) * q0.inverse).vector_part
    
    @staticmethod
    def AngleAxis(angle: float, axis: Vector3, deg: bool=True):
        """Initialise from axis and angle representation.
        Algorithm copied from pyquaternion's _from_axis_angle implementation
        and then modified to use this packages classes.
        """
        mag_sq = axis.sqrMagnitude
        if mag_sq == 0.0:
            raise ZeroDivisionError("Provided rotation axis has no length")
        
        # Ensure axis is in unit vector form
        axis = axis.normalized
        
        if deg:
            angle *= math.pi / 180

        theta = angle / 2.0
        r = math.cos(theta)
        i = axis * math.sin(theta)

        return Quat(r, i.x, i.y, i.z)
    #endregion

    #region euler angle related
    def GetEulerAngles(self, deg: bool=True, order: str='zxy') -> Vector3:
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/Quaternions.pdf
        assert all([letter in list('xyz') for letter in list(order)])
        p0 = self.w
        p1, p2, p3 = self.vector_part.transpose(order, inverse=False).ToList()
        i = Vector3(1,0,0)
        j = Vector3(0,1,0)
        k = Vector3(0,0,1)
        direction_map = {letter: direction for letter, direction in zip(list('xyz'), [i,j,k])}
        e3 = direction_map[order[2]]
        e2 = direction_map[order[1]]
        e1 = Quat.from_vector_part(e3) * Quat.from_vector_part(e2)
        e = getattr(e1, order[0])

        sin_yangle = 2 * (p0 * p2 + e * p1 * p3)
        if abs(sin_yangle) < 1: # Is this threshold small enough?
            y_angle = math.asin(sin_yangle)
            x_angle = math.atan2(
                2 * (p0 * p1 - e * p2 * p3),
                1 - 2 * (p1**2 + p2**2)
            )
            z_angle = math.atan2(
                2 * (p0 * p3 - e * p1 * p2),
                1 - 2 * (p2**2 + p3**2)
            )
        else:
            print("Warning: Encountered Singularity when calculating euler angles.")
            y_angle =  math.copysign(math.pi / 2, sin_yangle)
            z_angle = 0
            x_angle = math.atan2(p1, p0)
        if deg:
            x_angle = math.degrees(x_angle)
            y_angle = math.degrees(y_angle)
            z_angle = math.degrees(z_angle)
        
        return Vector3(x_angle, y_angle, z_angle).transpose(order, inverse=True)
    
    @property
    def eulerAngles(self) -> Vector3:
        return self.GetEulerAngles(deg=True)

    @staticmethod
    def EulerVector(euler: Vector3, deg: bool=True, order: str='zxy') -> Quat:
        assert all([letter in list('xyz') for letter in list(order)])
        # https://math.stackexchange.com/a/2975462
        # Note: Unity uses order='zxy'
        x0 = euler.x / 2
        y0 = euler.y / 2
        z0 = euler.z / 2
        if deg:
            x0 = math.radians(x0)
            y0 = math.radians(y0)
            z0 = math.radians(z0)
        qx = Quat(w=math.cos(x0), x=math.sin(x0), y=0, z=0)
        qy = Quat(w=math.cos(y0), x=0, y=math.sin(y0), z=0)
        qz = Quat(w=math.cos(z0), x=0, y=0, z=math.sin(z0))
        values = {'x': qx, 'y': qy, 'z': qz}
        result: Quat = None
        for letter in list(order):
            if result is None:
                result = values[letter]
            else:
                result = values[letter] * result
        return result
    
    @staticmethod
    def Euler(x: float, y: float, z: float, deg: bool=True, order: str='zxy') -> Quat:
        return Quat.EulerVector(Vector3(x,y,z), deg=deg, order=order)

    #endregion

    #region rotation matrix related
    @property
    def rotation_matrix(self) -> np.ndarray:
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        m00 = 1 - 2 * self.y**2 - 2 * self.z**2
        m01 = 2 * self.x * self.y - 2 * self.z * self.w
        m02 = 2 * self.x * self.z + 2 * self.y * self.w
        m10 = 2 * self.x * self.y + 2 * self.z * self.w
        m11 = 1 - 2 * self.x**2 - 2 * self.z**2
        m12 = 2 * self.y * self.z - 2 * self.x * self.w
        m20 = 2 * self.x * self.z - 2 * self.y * self.w
        m21 = 2 * self.y * self.z + 2 * self.x * self.w
        m22 = 1 - 2 * self.x**2 - 2 * self.y**2
        return np.array(
            [
                [m00, m01, m02],
                [m10, m11, m12],
                [m20, m21, m22]
            ]
        )

    @staticmethod
    def from_rotation_matrix(m: np.ndarray, diagonal: str='auto', neg_diagonal: bool=False) -> Quat:
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        if diagonal not in ['auto', 'w', 'x', 'y', 'z']:
            raise ValueError
        
        w, x, y, z = None, None, None, None
        if diagonal == 'auto':
            for d in ['w', 'x', 'y', 'z', 'fail']:
                if d == 'w':
                    inner = 1 + m[0,0] + m[1,1] + m[2,2]
                    if inner < 1e-12:
                        continue
                    w = 0.5 * inner**0.5
                    diagonal = d
                    break
                elif d == 'x':
                    inner = 1 + m[0,0] - m[1,1] - m[2,2]
                    if inner < 1e-12:
                        continue
                    x = 0.5 * inner**0.5
                    diagonal = d
                    break
                elif d == 'y':
                    inner = 1 + m[1,1] - m[0,0] - m[2,2]
                    if inner < 1e-12:
                        continue
                    y = 0.5 * inner**0.5
                    diagonal = d
                    break
                elif d == 'z':
                    inner = 1 + m[2,2] - m[0,0] - m[1,1]
                    if inner < 1e-12:
                        continue
                    z = 0.5 * inner**0.5
                    diagonal = d
                    break
                elif d == 'fail':
                    raise Exception('Failed to automatically find the correct diagnal for the quaternion calculation.')
                else:
                    raise Exception
            # print(f'Using diagonal: {diagonal}')

        # Note: There will always be 2 quaternion solutions to
        #       any given rotation matrix.
        #       Use neg_diagonal=True to get the other one.
        if diagonal == 'w':
            w = 0.5 * (1 + m[0,0] + m[1,1] + m[2,2])**0.5 if w is None else w
            w = w * -1 if neg_diagonal else w
            x = (m[2,1] - m[1,2]) / (4 * w)
            y = (m[0,2] - m[2,0]) / (4 * w)
            z = (m[1,0] - m[0,1]) / (4 * w)
            return Quat(w,x,y,z)
        elif diagonal == 'x':
            x = 0.5 * (1 + m[0,0] - m[1,1] - m[2,2])**0.5 if x is None else x
            x = x * -1 if neg_diagonal else x
            w = (m[2,1] - m[1,2]) / (4 * x)
            y = (m[1,0] + m[0,1]) / (4 * x)
            z = (m[0,2] + m[2,0]) / (4 * x)
            return Quat(w,x,y,z)
        elif diagonal == 'y':
            y = 0.5 * (1 + m[1,1] - m[0,0] - m[2,2])**0.5 if y is None else y
            y = y * -1 if neg_diagonal else y
            w = (m[0,2] - m[2,0]) / (4 * y)
            x = (m[1,0] + m[0,1]) / (4 * y)
            z = (m[2,1] + m[1,2]) / (4 * y)
            return Quat(w,x,y,z)
        elif diagonal == 'z':
            z = 0.5 * (1 + m[2,2] - m[0,0] - m[1,1])**0.5 if z is None else z
            z = z * -1 if neg_diagonal else z
            w = (m[1,0] - m[0,1]) / (4 * z)
            x = (m[0,2] - m[2,0]) / (4 * z)
            y = (m[2,1] - m[1,2]) / (4 * z)
            return Quat(w,x,y,z)
        else:
            raise Exception
    #endregion

    #region transformation matrix related
    @property
    def transformation_matrix(self) -> np.ndarray:
        mat = self.rotation_matrix
        mat = np.pad(mat, [(0, 1), (0, 1)], mode='constant', constant_values=0)
        mat[0:4, 3] = np.array([0, 0, 0, 1])
        return mat

    @staticmethod
    def from_transformation_matrix(m: np.ndarray) -> Quat:
        return Quat.from_rotation_matrix(m[:3, :3])
    
    def transform(self, v: Vector3) -> Vector3:
        print("Warning: Use of Quat.transform is discouraged. Please use Quat.rotate instead.")
        return Vector3.FromNumpy((self.transformation_matrix @ v.mat4.T).T.reshape(-1)[:3])
    #endregion
