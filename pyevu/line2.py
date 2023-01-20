from __future__ import annotations
from typing import TypeVar, Union
import copy
from .vector2 import Vector2
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
        
        refPoint = l0.p0
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
            r = other - self.p0
            if r.magnitude > 0:
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
    def unit_test():
        Line2.equality_test()
        Line2.parallel_test()
        Line2.distance_to_point_test()
        Line2.line_shortest_distance_test(verbose=False)
        Line2.intersects_test()
        Line2.slice_test()
        Line2.intersect_colinear_segments_test()
    #endregion

L = TypeVar('L', bound=Line2)
