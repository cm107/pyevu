from __future__ import annotations
from .vector2 import Vector2
from .vector3 import Vector3

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
    
    def __add__(self, other) -> Line2:
        if type(other) is Vector2:
            return Line2(self.p0 + other, self.p1 + other)
        else:
            raise TypeError
    
    def __radd__(self, other) -> Line2:
        if type(other) is Vector2:
            return Line2(self.p0 + other, self.p1 + other)
        else:
            raise TypeError
    
    def __sub__(self, other) -> Line2:
        if type(other) is Vector2:
            return Line2(self.p0 - other, self.p1 - other)
        else:
            raise TypeError

    def __iter__(self):
        return iter([self.p0, self.p1])

    @staticmethod
    def AreParallel(l0: Line2, l1: Line2, thresh: float=0.01) -> bool:
        """
        cos(angle) = Dot(a, b) / (a.mag * b.mag)
        Assuming a.mag == b.mag == 1
        a and b are parallel if Dot(a,b) is 1 or -1
        """
        m0 = (l0.p1 - l0.p0).normalized
        m1 = (l1.p1 - l1.p0).normalized
        dot = Vector2.Dot(m0, m1)
        return abs(abs(dot) - 1) < thresh

    @staticmethod
    def AreIntersecting(l0: Line2, l1: Line2, thresh: float=0.01) -> bool:
        """
        In 2D, lines intersect unless they are parallel.
        """
        return not Line2.AreParallel(l0, l1, thresh=thresh)          

    @staticmethod
    def ParallelShortestDistance(l0: Line2, l1: Line2, thresh: float=0.01) -> float:
        """
        If two lines are parallel, then the shortest distance between
        l0 and l1 is the same as the distance between an arbitrary point
        on l1 to l0.
        """
        return l0.get_distance_to_point(l1.p0, thresh=thresh)

    @staticmethod
    def ShortestDistance(l0: Line2, l1: Line2, thresh: float=0.01) -> float:
        if Line2.AreParallel(l0, l1, thresh=thresh):
            return Line2.ParallelShortestDistance(l0, l1, thresh=thresh)
        else:
            return 0 # The lines intersect

    @staticmethod
    def Intersection(l0: Line2, l1: Line2, thresh: float=0.01) -> Vector2:
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

    def get_distance_to_point(self, p: Vector2, thresh: float=0.01) -> float:
        """
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        """
        direction = (self.p1 - self.p0).normalized
        pVec = p - self.p0
        pProj = Vector2.Project(pVec, direction)
        perp = pVec - pProj
        return perp.magnitude
    
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

    def lineIsInFrontOfSegment(self, l: Line2, inclusive: bool=True) -> bool:
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
        
        if a0 < 0 and a1 > v.magnitude:
            # l completely encompasses self
            return True
        else:
            # l does not completely encompass self
            if inclusive:
                return 0 <= a0 <= v.magnitude or 0 <= a1 <= v.magnitude
            else:
                return 0 < a0 < v.magnitude or 0 < a1 < v.magnitude

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
    def unit_test():
        Line2.equality_test()
        Line2.parallel_test()
        Line2.distance_to_point_test()
        Line2.line_shortest_distance_test(verbose=False)
    #endregion