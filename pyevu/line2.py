from __future__ import annotations
from typing import TypeVar, Union
import copy
import numpy as np
from .vector2 import Vector2, Vector2Arr
from .interval import Interval

class Line2:
    def __init__(self, p0: Vector2, p1: Vector2):
        self.p0 = p0
        self.p1 = p1

    def __str__(self) -> str:
        return f'Line2({self.p0},{self.p1})'
    
    def __repr__(self) -> str:
        return self.__str__()

    def __key(self) -> tuple:
        return tuple([self.__class__] + list(self.__dict__.values()))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented
    
    def __add__(self, other) -> L:
        if type(other) is Vector2:
            return type(self)(self.p0 + other, self.p1 + other)
        else:
            raise TypeError
    
    def __radd__(self, other) -> L:
        if type(other) is Vector2:
            return type(self)(self.p0 + other, self.p1 + other)
        else:
            raise TypeError
    
    def __sub__(self, other) -> L:
        if type(other) is Vector2:
            return type(self)(self.p0 - other, self.p1 - other)
        else:
            raise TypeError

    def __iter__(self):
        return iter([self.p0, self.p1])

    def copy(self):
        return copy.deepcopy(self)

    def to_numpy(self) -> np.ndarray:
        return np.array(list(self.p0) + list(self.p1))

    @classmethod
    def from_numpy(self, arr: np.ndarray) -> Line2:
        return Line2(
            p0=Vector2.FromNumpy(arr[0:2]),
            p1=Vector2.FromNumpy(arr[2:4])
        )

    @property
    def midpoint(self) -> Vector2:
        return 0.5 * (self.p0 + self.p1)

    @classmethod
    def AreParallel(cls: type[L], l0: L, l1: L, thresh: float=1e-5) -> bool:
        """
        cos(angle) = Dot(a, b) / (a.mag * b.mag)
        Assuming a.mag == b.mag == 1
        a and b are parallel if Dot(a,b) is 1 or -1
        """
        m0 = (l0.p1 - l0.p0).normalized
        m1 = (l1.p1 - l1.p0).normalized
        dot = Vector2.Dot(m0, m1)
        return abs(abs(dot) - 1) < thresh

    @classmethod
    def AreIntersecting(cls: type[L], l0: L, l1: L, thresh: float=1e-5) -> bool:
        """
        In 2D, lines intersect unless they are parallel.
        """
        return not cls.AreParallel(l0, l1, thresh=thresh)          

    @classmethod
    def ParallelShortestDistance(cls: type[L], l0: L, l1: L, thresh: float=1e-5) -> float:
        """
        If two lines are parallel, then the shortest distance between
        l0 and l1 is the same as the distance between an arbitrary point
        on l1 to l0.
        """
        return l0.get_distance_to_point(l1.p0, thresh=thresh)

    @classmethod
    def ShortestDistance(cls: type[L], l0: L, l1: L, thresh: float=1e-5) -> float:
        if cls.AreParallel(l0, l1, thresh=thresh):
            return cls.ParallelShortestDistance(l0, l1, thresh=thresh)
        else:
            return 0 # The lines intersect

    @classmethod
    def Intersection(
        cls: type[L],
        l0: L, l1: L, thresh: float=1e-5
    ) -> Union[Vector2, None]:
        """
        Refer to https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
        """
        x1, y1 = list(l0.p0); x2, y2 = list(l0.p1); x3, y3 = list(l1.p0); x4, y4 = list(l1.p1)
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < thresh:
            return None

        det12 = x1 * y2 - y1 * x2; det34 = x3 * y4 - y3 * x4
        xNum = det12 * (x3 - x4) - (x1 - x2) * det34
        yNum = det12 * (y3 - y4) - (y1 - y2) * det34
        return Vector2(xNum / denom, yNum / denom)

    @classmethod
    def IntersectColinearSegments(
        cls: type[L],
        l0: L, l1: L, thresh: float=1e-5
    ) -> Union[L, None]:
        a = l0.p1 - l0.p0; b = l1.p1 - l1.p0
        dot = Vector2.Dot(a.normalized, b.normalized)
        isColinear = abs(abs(dot) - 1) < thresh
        if not isColinear:
            return None
        
        refPoint = l0.midpoint # TODO: This can probably be a problem sometimes.
        direction = a.normalized
        
        def projectPoint(p: Vector2) -> float:
            v = p - refPoint
            d = v.magnitude
            dot = Vector2.Dot(v.normalized, direction)
            if dot < 0:
                d *= -1
            return d
        
        def projectLine(l: L) -> Interval:
            val0 = projectPoint(l.p0)
            val1 = projectPoint(l.p1)
            if val0 < val1:
                return Interval(val0, val1)
            else:
                return Interval(val1, val0)
        
        i0 = projectLine(l0); i1 = projectLine(l1)
        i = Interval.Intersection(i0, i1)
        if i is None:
            return None # Colinear, but no intersection
        else:
            return cls(
                p0=refPoint + i.min * direction,
                p1=refPoint + i.max * direction
            )

    def get_distance_to_point(self, p: Vector2, thresh: float=1e-5) -> float:
        """
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        """
        direction = (self.p1 - self.p0).normalized
        pVec = p - self.p0
        pProj = Vector2.Project(pVec, direction)
        perp = pVec - pProj
        return perp.magnitude

    def intersects(
        self, other,
        thresh: float=1e-5,
        segment: bool=False, inclusive: bool=True
    ) -> bool:
        if type(other) is Vector2:
            if (other - self.p0).magnitude > (other - self.p1).magnitude:
                r = other - self.p0
                v = self.p1 - self.p0
            else:
                r = other - self.p1
                v = self.p0 - self.p1
            dot = Vector2.Dot(v.normalized, r.normalized)
            isColinear = abs(abs(dot) - 1) < thresh
            if not segment or not isColinear:
                return isColinear
            # is a segment and is colinear
            if dot < 0:
                return False
            if not inclusive:
                return 0 < r.magnitude < v.magnitude
            else:
                return 0 - thresh <= r.magnitude <= v.magnitude + thresh
        elif issubclass(type(other), Line2):
            if not type(self).AreIntersecting(self, other, thresh=thresh):
                return False
            intersectionPoint = type(self).Intersection(self, other, thresh=thresh)
            assert intersectionPoint is not None
            if not inclusive and intersectionPoint in list(self) + list(other):
                return False
            return self.intersects(
                intersectionPoint,
                thresh=thresh, segment=segment, inclusive=inclusive
            )
        else:
            raise TypeError

    def slice(self: L, line: L, thresh: float=1e-5, segment: bool=True) -> tuple[Union[L, None], Union[L, None]]:
        if self == line:
            return None, None
        intersectionPoint = type(self).Intersection(self, line, thresh=thresh)
        if intersectionPoint is None:
            return None, None
        intersectsSelf = self.intersects(intersectionPoint, thresh=thresh, segment=True, inclusive=True)
        intersectsLine = line.intersects(intersectionPoint, thresh=thresh, segment=True, inclusive=True)
        if segment:
            if not intersectsSelf or not intersectsLine:
                return None, None
        else:
            if not intersectsLine:
                return None, None

        if intersectionPoint == line.p0:
            return None, line.copy()
        elif intersectionPoint == line.p1:
            return line.copy(), None
        else:
            return type(self)(line.p0, intersectionPoint), type(self)(intersectionPoint, line.p1)

    def pointIsInFrontOfSegment(self, p: Vector2, inclusive: bool=True) -> bool:
        v = self.p1 - self.p0
        r = p - self.p0
        proj_r = Vector2.Project(r, v)
        a = proj_r.magnitude
        if Vector2.Dot(v, proj_r) < 0:
            a *= -1
        
        if inclusive:
            return 0 <= a <= v.magnitude
        else:
            return 0 < a < v.magnitude

    def lineIsInFrontOfSegment(self, l: L, inclusive: bool=True) -> bool:
        v = self.p1 - self.p0
        
        r0 = l.p0 - self.p0
        proj_r0 = Vector2.Project(r0, v)
        a0 = proj_r0.magnitude
        if Vector2.Dot(v, proj_r0) < 0:
            a0 *= -1
        
        r1 = l.p1 - self.p0
        proj_r1 = Vector2.Project(r1, v)
        a1 = proj_r1.magnitude
        if Vector2.Dot(v, proj_r1) < 0:
            a1 *= -1
        
        if not inclusive:
            if (
                (a0 < 0 and a1 > v.magnitude)
                or (a1 < 0 and a0 > v.magnitude)
            ):
                # l completely encompasses self
                return True
            else:
                # l does not completely encompass self
                return 0 < a0 < v.magnitude or 0 < a1 < v.magnitude
        else:
            if (
                (a0 <= 0 and a1 >= v.magnitude)
                or (a1 <= 0 and a0 >= v.magnitude)
            ):
                # l completely encompasses self
                return True
            else:
                # l does not completely encompass self
                return 0 <= a0 <= v.magnitude or 0 <= a1 <= v.magnitude

    #region Tests
    @staticmethod
    def equality_test():
        L = Line2; P = Vector2
        assert L(P(0,0), P(1,1)) == L(P(0,0), P(1,1))
        assert L(P(0,0), P(1,1)) != L(P(0,0), P(2,1))
        print("Equality Test Passed")
    
    @staticmethod
    def parallel_test():
        L = Line2; P = Vector2

        assert L.AreParallel(
            L(P(0,0), P(1,0)),
            L(P(0,2), P(1,2))
        )
        assert L.AreParallel(
            L(P(1,0), P(0,0)),
            L(P(0,2), P(1,2))
        )
        assert L.AreParallel(
            L(P(0,0), P(0,1)),
            L(P(1,0), P(1,1))
        )
        assert L.AreParallel(
            L(P(0,0), P(1,0)),
            L(P(1,2), P(0,2))
        )
        assert L.AreParallel(
            L(P(0,0), P(1,1)),
            L(P(-1,-1), P(-3,-3))
        )
        assert not L.AreParallel(
            L(P(0,0), P(1,1)),
            L(P(0,0), P(-1,1))
        )
        print("Parallel Test Passed")

    @staticmethod
    def distance_to_point_test():
        L = Line2; P = Vector2
        l = L(P(0,0), P(1,0))
        assert l.get_distance_to_point(P(0,1)) == 1
        assert l.get_distance_to_point(P(-1,0)) == 0
        print("Distance To Point Test Passed")

    @staticmethod
    def line_shortest_distance_test(verbose: bool=False):
        L = Line2; P = Vector2
        thresh = 0.001

        def get_flags_str(l0, l1, thresh) -> str:
            return ', '.join([
                f"{label}: {func(l0, l1, thresh)}"
                for label, func in [
                    ('Parallel', L.AreParallel),
                    ('Intersecting', L.AreIntersecting)
                ]
            ])

        l0 = L(P(0,0), P(1,0)); l1 = L(P(0,0), P(0,1)) + P(5,0)
        if verbose:
            print('Intersection')
            print(get_flags_str(l0, l1, thresh))
            print(f"{L.ShortestDistance(l0, l1, thresh=thresh)=}")
            print(f"{L.Intersection(l0, l1, thresh=thresh)=}")
        assert L.AreIntersecting(l0,l1,thresh)
        assert L.ShortestDistance(l0, l1, thresh=thresh) == 0
        assert L.Intersection(l0, l1, thresh=thresh) == Vector2(5,0)

        if verbose:
            print('')

        l0 = L(P(0,0), P(1,0)); l1 = L(P(0,0), P(1,0)) + P(0,5)
        if verbose:
            print('Parallel')
            print(get_flags_str(l0, l1, thresh))
            print(f"{L.ShortestDistance(l0, l1, thresh=thresh)=}")
            print(f"{L.Intersection(l0, l1, thresh=thresh)=}")
        assert L.AreParallel(l0,l1,thresh)
        assert L.ShortestDistance(l0, l1, thresh=thresh) == 5.0
        assert L.Intersection(l0, l1, thresh=thresh) is None

        print("Line Classification/Distance/Intersection Test Passed")

    @staticmethod
    def intersects_test():
        L = Line2; P = Vector2
        assert L(P(0,0), P(1,1)).intersects(P(0.5, 0.5))
        assert not L(P(0,0), P(1,1)).intersects(P(0.4, 0.5))
        assert L(P(0,0), P(1,1)).intersects(P(1.5, 1.5))
        assert not L(P(0,0), P(1,1)).intersects(P(1.5, 1.5), segment=True)
        assert L(P(0,0), P(1,1)).intersects(P(1, 1), segment=True, inclusive=True)
        assert not L(P(0,0), P(1,1)).intersects(P(1, 1), segment=True, inclusive=False)
        assert not L(P(0,0), P(1,1)).intersects(P(-1, -1), segment=True)

        assert L(P(0,0), P(1,1)).intersects(L(P(0,1), P(1,0)))
        assert L(P(0,0), P(1,1)).intersects(L(P(0,5), P(5,0)))
        assert not L(P(0,0), P(1,1)).intersects(L(P(0,5), P(5,0)), segment=True)
        assert L(P(0,0), P(1,1)).intersects(L(P(0.5,0.5), P(1,0)), segment=True, inclusive=True)
        assert not L(P(0,0), P(1,1)).intersects(L(P(0.5,0.5), P(1,0)), segment=True, inclusive=False)
        assert not L(P(0,0), P(1,1)).intersects(L(P(-0.5,-0.5), P(1,0)), segment=True)

        print("Intersection Test Passed")

    @staticmethod
    def slice_test():
        L = Line2; P = Vector2

        # Intersection in middle (Segment)
        l0 = L(P(0,1), P(1,0)); l1 = L(P(0,0), P(1,1)); p = P(0.5,0.5)
        assert l0.slice(l1) == (L(l1.p0, p), L(p, l1.p1))

        # Intersection at start (Segment)
        l0 = L(P(-1,1), P(1,-1)); l1 = L(P(0,0), P(1,1))
        assert l0.slice(l1) == (None, l1)

        # Intersection at end (Segment)
        l0 = L(P(0,2), P(2,0)); l1 = L(P(0,0), P(1,1))
        assert l0.slice(l1) == (l1, None)

        # No intersection (Segment)
        l0 = L(P(0,5), P(5,0)); l1 = L(P(0,0), P(1,1))
        assert l0.slice(l1) == (None, None)

        # Intersection in middle (Line)
        l0 = L(P(0,5), P(1,4)); l1 = L(P(0,0), P(5,5)); p = P(2.5,2.5)
        assert l0.slice(l1, segment=False) == (L(l1.p0, p), L(p, l1.p1))

        # Intersection at start (Line)
        l0 = L(P(-5,5), P(-4,4)); l1 = L(P(0,0), P(5,5))
        assert l0.slice(l1, segment=False) == (None, l1)

        # Intersection at end (Line)
        l0 = L(P(0,10), P(1,9)); l1 = L(P(0,0), P(5,5))
        assert l0.slice(l1, segment=False) == (l1, None)

        # No Intersection (Line)
        l0 = L(P(0,11), P(1,10)); l1 = L(P(0,0), P(5,5))
        assert l0.slice(l1, segment=False) == (None, None)

        print("Slice Test Passed")

    @staticmethod
    def intersect_colinear_segments_test():
        L = Line2; P = Vector2

        def samePoint(p0: P, p1: P, thresh: float=1e-5) -> bool:
            sameX = abs(p1.x - p0.x) < thresh
            sameY = abs(p1.y - p0.x) < thresh
            return sameX and sameY
        
        def sameLine(l0: L, l1: L, thresh: float=1e-5) -> bool:
            sameP0 = samePoint(l0.p0, l1.p0, thresh=thresh)
            sameP1 = samePoint(l0.p1, l1.p1, thresh=thresh)
            return sameP0 and sameP1

        # Not Colinear
        l0 = L(P(0,0), P(1,1)); l1 = L(P(0,1), P(1,0))
        assert L.IntersectColinearSegments(l0, l1) is None

        # Colinear, but not intersecting
        l0 = L(P(0,0), P(1,1)); l1 = L(P(2,2), P(3,3))
        assert L.IntersectColinearSegments(l0, l1) is None

        # Colinear and intersecting
        l0 = L(P(0,0), P(1,1)); l1 = L(P(0.5,0.5), P(3,3))
        l = L(P(0.5,0.5), P(1,1))
        assert sameLine(L.IntersectColinearSegments(l0, l1), l)

        # Colinear and completely enclosed
        l0 = L(P(0,0), P(1,1)); l1 = L(P(0.5,0.5), P(0.7,0.7))
        l = L(P(0.5,0.5), P(0.7,0.7))
        result = L.IntersectColinearSegments(l0, l1)
        assert sameLine(result, l)

        # Colinear but intersecting at just one point
        l0 = L(P(0,0), P(1,1)); l1 = L(P(1,1), P(2,2))
        l = L(P(1,1), P(1,1)) # This could be useful?
        assert sameLine(L.IntersectColinearSegments(l0, l1), l)

        print("Intersect Colinear Segments Test Passed")

    @staticmethod
    def numpy_test():
        l = Line2(Vector2(1,2), Vector2(3,4))
        assert all(l.to_numpy() == Line2.from_numpy(l.to_numpy()).to_numpy())
        print("Numpy test passed")

    @staticmethod
    def unit_test():
        Line2.equality_test()
        Line2.parallel_test()
        Line2.distance_to_point_test()
        Line2.line_shortest_distance_test(verbose=False)
        Line2.intersects_test()
        Line2.slice_test()
        Line2.intersect_colinear_segments_test()
        Line2.numpy_test()
    #endregion

L = TypeVar('L', bound=Line2)

import numpy.typing as npt
from typing import Annotated, Literal, Generator

DType = TypeVar("DType", bound=np.generic)
ArrayNx4 = Annotated[npt.NDArray[DType], Literal["N", 4]]

class Line2Arr:
    dtype = np.float64

    def __init__(self, p0: Vector2Arr, p1: Vector2Arr):
        self.p0 = p0
        self.p1 = p1
    
    def __str__(self) -> str:
        return str(self.to_numpy())

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.p0)

    def __add__(self, other) -> LA:
        if type(other) is Vector2:
            return type(self)(
                p0=self.p0 + other,
                p1=self.p1 + other
            )
        elif type(other) is Vector2Arr:
            return type(self).from_numpy(
                self.to_numpy() + np.tile(other.to_numpy(), 2)
            )
        else:
            raise TypeError
    
    def __radd__(self, other) -> LA:
        return self.__add__(other)

    def __sub__(self, other) -> LA:
        if type(other) is Vector2:
            return type(self)(
                p0=self.p0 - other,
                p1=self.p1 - other
            )
        elif type(other) is Vector2Arr:
            return type(self).from_numpy(
                self.to_numpy() - np.tile(other.to_numpy(), 2)
            )
        else:
            raise TypeError

    def __getitem__(self, idx: int) -> Line2:
        if type(idx) is not int:
            raise TypeError
        return Line2(self.p0[idx], self.p1[idx])

    def __iter__(self) -> Generator[Line2]:
        for i in range(len(self)):
            yield self[i]

    def to_numpy(self) -> ArrayNx4:
        return np.hstack([self.p0.to_numpy(), self.p1.to_numpy()]).astype(type(self).dtype)

    @classmethod
    def from_numpy(cls, arr: ArrayNx4) -> Line2Arr:
        return Line2Arr(
            p0=Vector2Arr.from_numpy(arr[:, 0:2]),
            p1=Vector2Arr.from_numpy(arr[:, 2:4])
        )
    
    def to_lines(self) -> list[Line2]:
        return [
            Line2(p0, p1)
            for p0, p1 in zip(self.p0.to_vectors(), self.p1.to_vectors())
        ]
    
    @classmethod
    def from_lines(cls, lines: list[Line2]) -> Line2Arr:
        return Line2Arr(
            p0=Vector2Arr.from_vectors([line.p0 for line in lines]),
            p1=Vector2Arr.from_vectors([line.p1 for line in lines])
        )
    
    @property
    def midpoint(self) -> Vector2Arr:
        # return 0.5 * (self.p0 + self.p1)
        raise NotImplementedError

    @classmethod
    def AreParallel(cls: type[LA], l0: LA, l1: LA, thresh: float=1e-5) -> npt.NDArray[np.bool_]:
        """
        cos(angle) = Dot(a, b) / (a.mag * b.mag)
        Assuming a.mag == b.mag == 1
        a and b are parallel if Dot(a,b) is 1 or -1
        """
        # m0 = (l0.p1 - l0.p0).normalized
        # m1 = (l1.p1 - l1.p0).normalized
        # dot = Vector2.Dot(m0, m1)
        # return abs(abs(dot) - 1) < thresh

        m0 = (l0.p1 - l0.p0).normalized
        m1 = (l1.p1 - l1.p0).normalized
        dot = Vector2Arr.Dot(m0, m1)
        result = np.abs(np.abs(dot) - 1) < thresh
        return result

    @classmethod
    def AreIntersecting(cls: type[LA], l0: LA, l1: LA, thresh: float=1e-5) -> npt.NDArray[np.bool_]:
        """
        In 2D, lines intersect unless they are parallel.
        """
        # return not cls.AreParallel(l0, l1, thresh=thresh)
        raise NotImplementedError

    @classmethod
    def ParallelShortestDistance(cls: type[LA], l0: LA, l1: LA, thresh: float=1e-5) -> np.ndarray:
        """
        If two lines are parallel, then the shortest distance between
        l0 and l1 is the same as the distance between an arbitrary point
        on l1 to l0.
        """
        # return l0.get_distance_to_point(l1.p0, thresh=thresh)
        return l0.get_distance_to_point(l1.p0, thresh=thresh)

    @classmethod
    def ShortestDistance(cls: type[LA], l0: LA, l1: LA, thresh: float=1e-5) -> np.ndarray:
        # if cls.AreParallel(l0, l1, thresh=thresh):
        #     return cls.ParallelShortestDistance(l0, l1, thresh=thresh)
        # else:
        #     return 0 # The lines intersect
        raise NotImplementedError

    @classmethod
    def Intersection(
        cls: type[LA],
        l0: LA, l1: LA, thresh: float=1e-5
    ) -> Vector2Arr:
        """
        Refer to https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
        """
        # x1, y1 = list(l0.p0); x2, y2 = list(l0.p1); x3, y3 = list(l1.p0); x4, y4 = list(l1.p1)
        # denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        # if abs(denom) < thresh:
        #     return None

        # det12 = x1 * y2 - y1 * x2; det34 = x3 * y4 - y3 * x4
        # xNum = det12 * (x3 - x4) - (x1 - x2) * det34
        # yNum = det12 * (y3 - y4) - (y1 - y2) * det34
        # return Vector2(xNum / denom, yNum / denom)
        raise NotImplementedError

    @classmethod
    def IntersectColinearSegments(
        cls: type[LA],
        l0: LA, l1: LA, thresh: float=1e-5
    ) -> LA:
        # a = l0.p1 - l0.p0; b = l1.p1 - l1.p0
        # dot = Vector2.Dot(a.normalized, b.normalized)
        # isColinear = abs(abs(dot) - 1) < thresh
        # if not isColinear:
        #     return None
        
        # refPoint = l0.midpoint # TODO: This can probably be a problem sometimes.
        # direction = a.normalized
        
        # def projectPoint(p: Vector2) -> float:
        #     v = p - refPoint
        #     d = v.magnitude
        #     dot = Vector2.Dot(v.normalized, direction)
        #     if dot < 0:
        #         d *= -1
        #     return d
        
        # def projectLine(l: L) -> Interval:
        #     val0 = projectPoint(l.p0)
        #     val1 = projectPoint(l.p1)
        #     if val0 < val1:
        #         return Interval(val0, val1)
        #     else:
        #         return Interval(val1, val0)
        
        # i0 = projectLine(l0); i1 = projectLine(l1)
        # i = Interval.Intersection(i0, i1)
        # if i is None:
        #     return None # Colinear, but no intersection
        # else:
        #     return cls(
        #         p0=refPoint + i.min * direction,
        #         p1=refPoint + i.max * direction
        #     )
        raise NotImplementedError

    def get_distance_to_point(self, p: Union[Vector2, Vector2Arr], thresh: float=1e-5) -> np.ndarray:
        """
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        """
        # direction = (self.p1 - self.p0).normalized
        # pVec = p - self.p0
        # pProj = Vector2.Project(pVec, direction)
        # perp = pVec - pProj
        # return perp.magnitude

        if issubclass(type(p), Vector2):
            p = np.tile(p.ToNumpy(), (len(self),1))
            p = Vector2Arr.from_numpy(p)
        elif issubclass(type(p), Vector2Arr):
            pass
        else:
            raise TypeError
        
        direction = (self.p1 - self.p0).normalized
        pVec = p - self.p0
        pProj = Vector2Arr.Project(pVec, direction)
        perp: Vector2Arr = pVec - pProj
        return perp.magnitude

    def intersects(
        self, other,
        thresh: float=1e-5,
        segment: bool=False, inclusive: bool=True
    ) -> npt.NDArray[np.bool_]:
        # if type(other) is Vector2:
        #     if (other - self.p0).magnitude > (other - self.p1).magnitude:
        #         r = other - self.p0
        #         v = self.p1 - self.p0
        #     else:
        #         r = other - self.p1
        #         v = self.p0 - self.p1
        #     dot = Vector2.Dot(v.normalized, r.normalized)
        #     isColinear = abs(abs(dot) - 1) < thresh
        #     if not segment or not isColinear:
        #         return isColinear
        #     # is a segment and is colinear
        #     if dot < 0:
        #         return False
        #     if not inclusive:
        #         return 0 < r.magnitude < v.magnitude
        #     else:
        #         return 0 - thresh <= r.magnitude <= v.magnitude + thresh
        # elif issubclass(type(other), Line2):
        #     if not type(self).AreIntersecting(self, other, thresh=thresh):
        #         return False
        #     intersectionPoint = type(self).Intersection(self, other, thresh=thresh)
        #     assert intersectionPoint is not None
        #     if not inclusive and intersectionPoint in list(self) + list(other):
        #         return False
        #     return self.intersects(
        #         intersectionPoint,
        #         thresh=thresh, segment=segment, inclusive=inclusive
        #     )
        # else:
        #     raise TypeError
        raise NotImplementedError

    def slice(
        self: LA, line: Union[L, LA], thresh: float=1e-5, segment: bool=True
    ) -> tuple[LA, LA]:
        # if self == line:
        #     return None, None
        # intersectionPoint = type(self).Intersection(self, line, thresh=thresh)
        # if intersectionPoint is None:
        #     return None, None
        # intersectsSelf = self.intersects(intersectionPoint, thresh=thresh, segment=True, inclusive=True)
        # intersectsLine = line.intersects(intersectionPoint, thresh=thresh, segment=True, inclusive=True)
        # if segment:
        #     if not intersectsSelf or not intersectsLine:
        #         return None, None
        # else:
        #     if not intersectsLine:
        #         return None, None

        # if intersectionPoint == line.p0:
        #     return None, line.copy()
        # elif intersectionPoint == line.p1:
        #     return line.copy(), None
        # else:
        #     return type(self)(line.p0, intersectionPoint), type(self)(intersectionPoint, line.p1)
        raise NotImplementedError

    def pointIsInFrontOfSegment(
        self, p: Union[Vector2, Vector2Arr], inclusive: bool=True
    ) -> npt.NDArray[np.bool_]:
        # v = self.p1 - self.p0
        # r = p - self.p0
        # proj_r = Vector2.Project(r, v)
        # a = proj_r.magnitude
        # if Vector2.Dot(v, proj_r) < 0:
        #     a *= -1
        
        # if inclusive:
        #     return 0 <= a <= v.magnitude
        # else:
        #     return 0 < a < v.magnitude
        raise NotImplementedError

    def lineIsInFrontOfSegment(
        self, l: Union[L, LA], inclusive: bool=True
    ) -> npt.NDArray[np.bool_]:
        if issubclass(type(l), Line2):
            l = np.tile(l.to_numpy(), (len(self),1))
            l = Line2Arr.from_numpy(l)
        elif issubclass(type(l), Line2Arr):
            pass
        else:
            raise TypeError
        
        # v = self.p1 - self.p0
        v = self.p1 - self.p0
        # print(f"{v=}")
        # if True:
        #     for i, (_p1, _p0) in enumerate(zip(self.p1, self.p0)):
        #         assert v[i] == _p1 - _p0
        
        # r0 = l.p0 - self.p0
        # proj_r0 = Vector2.Project(r0, v)
        # a0 = proj_r0.magnitude
        # if Vector2.Dot(v, proj_r0) < 0:
        #     a0 *= -1
        r0 = l.p0 - self.p0
        proj_r0 = Vector2Arr.Project(r0, v)
        a0: np.ndarray = proj_r0.magnitude
        a0NegMask: npt.NDArray[np.bool_] = Vector2Arr.Dot(v, proj_r0) < 0
        a0[a0NegMask] = a0[a0NegMask] * -1
        # print(f"{a0=}")
        # if True:
        #     for i, (_l, _v, _p0) in enumerate(zip(l, v, self.p0)):
        #         _r0 = _l.p0 - _p0
        #         _proj_r0 = Vector2.Project(_r0, _v)
        #         _a0 = _proj_r0.magnitude
        #         _a0NegMask = Vector2.Dot(_v, _proj_r0) < 0
        #         _a0 = _a0 * -1 if _a0NegMask else _a0
        #         assert round(a0[i],4) == round(_a0,4), f"{i=}, {a0[i]=}, {_a0=}"

        # r1 = l.p1 - self.p0
        # proj_r1 = Vector2.Project(r1, v)
        # a1 = proj_r1.magnitude
        # if Vector2.Dot(v, proj_r1) < 0:
        #     a1 *= -1
        r1 = l.p1 - self.p0
        proj_r1 = Vector2Arr.Project(r1, v)
        a1: np.ndarray = proj_r1.magnitude
        a1NegMask: npt.NDArray[np.bool_] = Vector2Arr.Dot(v, proj_r1) < 0
        a1[a1NegMask] = a1[a1NegMask] * -1
        # print(f"{a1=}")
        # if True:
        #     for i, (_l, _v, _p0) in enumerate(zip(l, v, self.p0)):
        #         _r1 = _l.p1 - _p0
        #         _proj_r1 = Vector2.Project(_r1, _v)
        #         _a1 = _proj_r1.magnitude
        #         _a1NegMask = Vector2.Dot(_v, _proj_r1) < 0
        #         _a1 = _a1 * -1 if _a1NegMask else _a1
        #         assert round(a1[i],4) == round(_a1,4), f"{i=}, {a1[i]=}, {_a1=}"

        # if not inclusive:
        #     if (
        #         (a0 < 0 and a1 > v.magnitude)
        #         or (a1 < 0 and a0 > v.magnitude)
        #     ):
        #         # l completely encompasses self
        #         return True
        #     else:
        #         # l does not completely encompass self
        #         return 0 < a0 < v.magnitude or 0 < a1 < v.magnitude
        # else:
        #     if (
        #         (a0 <= 0 and a1 >= v.magnitude)
        #         or (a1 <= 0 and a0 >= v.magnitude)
        #     ):
        #         # l completely encompasses self
        #         return True
        #     else:
        #         # l does not completely encompass self
        #         return 0 <= a0 <= v.magnitude or 0 <= a1 <= v.magnitude
        
        vMag = v.magnitude
        if not inclusive:
            encompassSelfMask0 = np.logical_and(
                a0 < 0,
                a1 > vMag
            )
            encompassSelfMask1 = np.logical_and(
                a1 < 0,
                a0 > vMag
            )
            encompassSelfMask = np.logical_or(
                encompassSelfMask0,
                encompassSelfMask1
            )
            partialEncompassMask = np.logical_or(
                np.logical_and(0 < a0, a0 < vMag),
                np.logical_and(0 < a1, a1 < vMag)
            )
            result = np.logical_or(
                encompassSelfMask,
                partialEncompassMask
            )
        else:
            encompassSelfMask0 = np.logical_and(
                a0 <= 0,
                a1 >= vMag
            )
            encompassSelfMask1 = np.logical_and(
                a1 <= 0,
                a0 >= vMag
            )
            encompassSelfMask = np.logical_or(
                encompassSelfMask0,
                encompassSelfMask1
            )
            # print(f"{encompassSelfMask=}")
            partialEncompassMask = np.logical_or(
                np.logical_and(0 <= a0, a0 <= vMag),
                np.logical_and(0 <= a1, a1 <= vMag)
            )
            # print(f"{partialEncompassMask=}")
            result = np.logical_or(
                encompassSelfMask,
                partialEncompassMask
            )
        # print(f"{vMag=}")
        # print(f"{a0=}")
        # print(f"{a1=}")
        # print(f"{encompassSelfMask=}")
        # print(f"{partialEncompassMask=}")
        # print(f"{result=}")
        # if True:
        #     for i, (_a0, _a1, _v) in enumerate(zip(a0, a1, v)):
        #         if not inclusive:
        #             if (
        #                 (_a0 < 0 and _a1 > _v.magnitude)
        #                 or (_a1 < 0 and _a0 > _v.magnitude)
        #             ):
        #                 # l completely encompasses self
        #                 _result = True
        #             else:
        #                 # l does not completely encompass self
        #                 _result = 0 < _a0 < _v.magnitude or 0 < _a1 < _v.magnitude
        #         else:
        #             if (
        #                 (_a0 <= 0 and _a1 >= _v.magnitude)
        #                 or (_a1 <= 0 and _a0 >= _v.magnitude)
        #             ):
        #                 # l completely encompasses self
        #                 _result = True
        #             else:
        #                 # l does not completely encompass self
        #                 _result = 0 <= _a0 <= _v.magnitude or 0 <= _a1 <= _v.magnitude
        #         assert result[i] == _result, f"{i=}, {_a0=}, {_a1=}, {_v.magnitude=}, {result[i]=}, {_result=}"
        return result

    @staticmethod
    def general_test():
        aVecs = [
            Vector2(0,0),
            Vector2(1,2),
            Vector2(3,4),
            Vector2(5,6),
            Vector2(7,8),
            Vector2(9,10),
        ]
        bVecs = [
            Vector2(0,0),
            Vector2(11,12),
            Vector2(13,14),
            Vector2(15,16),
            Vector2(17,18),
            Vector2(19,20),
        ][::-1]
        cVecs = [
            Vector2(26,97),
            Vector2(43,12),
            Vector2(58,145),
            Vector2(13,567),
            Vector2(23,67),
            Vector2(12,79),
        ]
        dVecs = [
            Vector2(0,0),
            Vector2(32,68),
            Vector2(23,89),
            Vector2(15,3),
            Vector2(2,79),
            Vector2(59,10),
        ][::-1]
        larr0 = Line2Arr(
            Vector2Arr.from_vectors(aVecs),
            Vector2Arr.from_vectors(bVecs)
        )
        larr1 = Line2Arr(
            Vector2Arr.from_vectors(cVecs),
            Vector2Arr.from_vectors(dVecs)
        )

        assert (
            larr0.to_numpy() == Line2Arr.from_numpy(
                larr0.to_numpy()
            ).to_numpy()
        ).all()

        def same_vector(v0: Vector2, v1: Vector2, thresh: float=1e-5) -> bool:
            diff = v1 - v0
            return abs(diff.x) < thresh and abs(diff.y) < thresh
        
        def same_line(l0: Line2, l1: Line2, thresh: float=1e-5) -> bool:
            return same_vector(l0.p0, l1.p0, thresh) and same_vector(l0.p1, l1.p1, thresh)

        arr = larr0 + Vector2(4, 5)
        for i, l in enumerate(larr0.to_lines()):
            assert same_line(arr.to_lines()[i], l + Vector2(4, 5))

        arr = larr0 + larr1.p0
        for i, (l, p) in enumerate(zip(larr0.to_lines(), larr1.p0.to_vectors())):
            assert same_line(arr.to_lines()[i], l + p)
        
        print("General Test Passed")

    @staticmethod
    def line_is_in_front_of_segment_test():
        from itertools import permutations
        import random
        # Vector2Arr.dtype = np.float128
        # Line2Arr.dtype = np.float128
        vecs = [
            Vector2(0, 0), Vector2(100, 200), Vector2(50, 20), Vector2(1000, 3000),
            Vector2(-40, 20), Vector2(-100, -100), Vector2(-60, -7000), Vector2(-5000, -5000)
        ]

        # k = 3; sliceSize = 50
        k = 0; sliceSize = 10000
        perms = list(permutations(list(range(len(vecs))), r=4))
        random.Random(123).shuffle(perms)
        perms = perms[k*sliceSize:(k+1)*sliceSize]
        # print(f"{perms=}")
        larr0 = Line2Arr(
            Vector2Arr.from_vectors([vecs[perm[0]] for perm in perms]),
            Vector2Arr.from_vectors([vecs[perm[1]] for perm in perms])
        )
        larr1 = Line2Arr(
            Vector2Arr.from_vectors([vecs[perm[2]] for perm in perms]),
            Vector2Arr.from_vectors([vecs[perm[3]] for perm in perms])
        )
        result = larr0.lineIsInFrontOfSegment(larr1)
        # print(f"{result=}")
        count = 0; total = 0
        for i, (line0, line1) in enumerate(zip(larr0.to_lines(), larr1.to_lines())):
            actual = line0.lineIsInFrontOfSegment(line1)
            # assert result[i] == actual, \
            #     f"{i=}, {line0=}, {line1=}, {result[i]=}, {actual=}"
            if result[i] == actual:
                count += 1
            total += 1
        # print(f"{count}/{total}: {count/total}")
        assert count / total > 0.99 # Note: Some differences are due to rounding.


        # result = larr0.lineIsInFrontOfSegment(larr1)

        print("Line Is In Front Of Segment Test Pass")

    @staticmethod
    def parallel_test():
        L = Line2; P = Vector2; LA = Line2Arr
        l0_list: list[Line2] = []; l1_list: list[Line2] = []
        expected_result: list[bool] = []

        l0_list.append(L(P(0,0), P(1,0)))
        l1_list.append(L(P(0,2), P(1,2)))
        expected_result.append(True)

        l0_list.append(L(P(1,0), P(0,0)))
        l1_list.append(L(P(0,2), P(1,2)))
        expected_result.append(True)

        l0_list.append(L(P(0,0), P(0,1)))
        l1_list.append(L(P(1,0), P(1,1)))
        expected_result.append(True)

        l0_list.append(L(P(0,0), P(1,0)))
        l1_list.append(L(P(1,2), P(0,2)))
        expected_result.append(True)

        l0_list.append(L(P(0,0), P(1,1)))
        l1_list.append(L(P(-1,-1), P(-3,-3)))
        expected_result.append(True)

        l0_list.append(L(P(0,0), P(1,1)))
        l1_list.append(L(P(0,0), P(-1,1)))
        expected_result.append(False)

        l0 = Line2Arr.from_lines(l0_list)
        l1 = Line2Arr.from_lines(l1_list)
        actual_result = Line2Arr.AreParallel(l0, l1)
        for i, (_l0, _l1, _actual, _expected) in enumerate(
            zip(l0, l1, actual_result, expected_result)
        ):
            assert _actual == _expected
            assert _actual == Line2.AreParallel(_l0, _l1)

        print("Parallel Test Passed")

    @staticmethod
    def distance_to_point_test():
        L = Line2; P = Vector2; PA = Vector2Arr; LA = Line2Arr
        larr = LA.from_lines(
            [
                L(P(0,0), P(1,0)),
                L(P(0,0), P(0,1)),
                L(P(32,76), P(-7,15)),
                L(P(-6,30), P(24,100)),
            ]
        )
        parr = PA.from_vectors(
            [
                P(36, 71),
                P(10, -1020),
                P(-10,77),
                P(360, 710)
            ]
        )

        distances = larr.get_distance_to_point(parr)
        for i, (l, p) in enumerate(zip(larr, parr)):
            assert distances[i] == l.get_distance_to_point(p)
        
        for _p in parr:
            distances = larr.get_distance_to_point(_p)
            for i, l in enumerate(larr):
                expected = l.get_distance_to_point(_p)
                assert distances[i] == expected, \
                    f"{i=}, {distances[i]=}, {expected=}"
        
        print("Distance To Point Test Passed")

    @staticmethod
    def unit_test():
        Line2Arr.general_test()
        Line2Arr.line_is_in_front_of_segment_test()
        Line2Arr.parallel_test()
        Line2Arr.distance_to_point_test()


LineArrVar = Union[Line2Arr, ArrayNx4]
LA = TypeVar('LA', bound=Line2Arr)