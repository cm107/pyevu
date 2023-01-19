from __future__ import annotations
from .vector3 import Vector3

class Line3:
    def __init__(self, p0: Vector3, p1: Vector3):
        self.p0 = p0
        self.p1 = p1

    def __str__(self) -> str:
        return f'Line3({self.p0},{self.p1})'
    
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
    
    def __add__(self, other) -> Line3:
        if type(other) is Vector3:
            return Line3(self.p0 + other, self.p1 + other)
        else:
            raise TypeError
    
    def __radd__(self, other) -> Line3:
        if type(other) is Vector3:
            return Line3(self.p0 + other, self.p1 + other)
        else:
            raise TypeError
    
    def __sub__(self, other) -> Line3:
        if type(other) is Vector3:
            return Line3(self.p0 - other, self.p1 - other)
        else:
            raise TypeError

    def __iter__(self):
        return iter([self.p0, self.p1])

    @staticmethod
    def AreParallel(l0: Line3, l1: Line3, thresh: float=0.01) -> bool:
        """
        cos(angle) = Dot(a, b) / (a.mag * b.mag)
        Assuming a.mag == b.mag == 1
        a and b are parallel if Dot(a,b) is 1 or -1
        """
        m0 = (l0.p1 - l0.p0).normalized
        m1 = (l1.p1 - l1.p0).normalized
        dot = Vector3.Dot(m0, m1)
        return abs(abs(dot) - 1) < thresh

    @staticmethod
    def AreCoplanar(l0: Line3, l1: Line3, thresh: float=0.01) -> bool:
        """
        Two lines are coplanar if any line that intersects both lines
        is orthogonal to the cross product of the two lines.
        Dot((p2 - p1), Cross(m1,m2)) == 0
        
        Refer to https://www.toppr.com/guides/maths/three-dimensional-geometry/coplanarity-of-two-lines/
        """
        p0: Vector3 = l0.p0; m0: Vector3 = (l0.p1 - l0.p0).normalized
        p1: Vector3 = l1.p0; m1: Vector3 = (l1.p1 - l1.p0).normalized
        p_diff = p1 - p0
        if p_diff.magnitude < thresh:
            p1 = l1.p1
            p_diff = p1 - p0
        m_cross = Vector3.Cross(m0, m1)
        return abs(Vector3.Dot(p_diff, m_cross)) < thresh

    @staticmethod
    def AreIntersecting(l0: Line3, l1: Line3, thresh: float=0.01) -> bool:
        """
        Two non-parallel lines intersect if and only if they are coplanar.
        Refer to https://math.stackexchange.com/a/697278
        """
        if Line3.AreParallel(l0, l1, thresh=thresh):
            return False
        else:
            return Line3.AreCoplanar(l0, l1, thresh=thresh)            

    @staticmethod
    def AreSkew(l0: Line3, l1: Line3, thresh: float=0.01) -> bool:
        """
        Skew lines are a pair of lines that are non-intersecting,
        non-parallel, and non-coplanar.
        """
        return (
           not Line3.AreParallel(l0, l1, thresh=thresh)
           and not Line3.AreCoplanar(l0, l1, thresh=thresh) 
        )

    @staticmethod
    def SkewShortestDistance(l0: Line3, l1: Line3, thresh: float=0.01) -> float:
        """
        Finds the shortest distance between two skew lines.
        A set of lines are skew if they do not intersect each
        other at any point and are not parallel.
        
        Note: This algorithm assumes that the inputted lines are skew,
        and does not check.

        Refer to https://byjus.com/jee/shortest-distance-between-two-lines/
        """
        p0: Vector3 = l0.p0; m0: Vector3 = (l0.p1 - l0.p0).normalized
        p1: Vector3 = l1.p0; m1: Vector3 = (l1.p1 - l1.p0).normalized
        p_diff = p1 - p0
        if p_diff.magnitude < thresh:
            p1 = l1.p1
            p_diff = p1 - p0
        m_cross = Vector3.Cross(m0, m1)
        return abs(Vector3.Dot(m_cross, p_diff) / m_cross.magnitude)

    @staticmethod
    def ParallelShortestDistance(l0: Line3, l1: Line3, thresh: float=0.01) -> float:
        """
        If two lines are parallel, then the shortest distance between
        l0 and l1 is the same as the distance between an arbitrary point
        on l1 to l0.
        """
        return l0.get_distance_to_point(l1.p0, thresh=thresh)

    @staticmethod
    def ShortestDistance(l0: Line3, l1: Line3, thresh: float=0.01) -> float:
        if Line3.AreParallel(l0, l1, thresh=thresh):
            return Line3.ParallelShortestDistance(l0, l1, thresh=thresh)
        else:
            if Line3.AreCoplanar(l0, l1, thresh=thresh):
                return 0 # The lines intersect
            else:
                # not parallel, not intersecting, not coplanar
                # This is the definition of skew lines.
                return Line3.SkewShortestDistance(l0, l1, thresh=thresh)

    @staticmethod
    def Intersection(l0: Line3, l1: Line3, thresh: float=0.01) -> Vector3:
        """
        Refer to https://math.stackexchange.com/questions/270767/find-intersection-of-two-3d-lines/271366
        """
        if not Line3.AreIntersecting(l0, l1, thresh=thresh):
            return None

        p0: Vector3 = l0.p0; m0: Vector3 = l0.p1 - l0.p0
        p1: Vector3 = l1.p0; m1: Vector3 = l1.p1 - l1.p0
        p_diff = p1 - p0
        if p_diff.magnitude < thresh:
            p1 = l1.p1
            p_diff = p1 - p0
        
        f_cross_g = Vector3.Cross(m1, p_diff)
        f_cross_e = Vector3.Cross(m1, m0)

        if f_cross_e.magnitude == 0:
            return None

        M = p0

        if Vector3.Dot(f_cross_g, f_cross_e) > 0:
            M += (f_cross_g.magnitude / f_cross_e.magnitude) * m0
        else:
            M -= (f_cross_g.magnitude / f_cross_e.magnitude) * m0
        return M

    def get_distance_to_point(self, p: Vector3, thresh: float=0.01) -> float:
        """
        https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        """
        # a = self.p0
        # ap = p - a
        # if ap.magnitude < thresh:
        #     a = self.p1
        #     ap = p - a
        # n = self.p1 - self.p0
        # return Vector3.Cross(ap, n).magnitude / n.magnitude

        direction = (self.p1 - self.p0).normalized
        pVec = p - self.p0
        pProj = Vector3.Project(pVec, direction)
        perp = pVec - pProj
        return perp.magnitude

    #region Tests
    @staticmethod
    def equality_test():
        L = Line3; P = Vector3
        assert L(P(0,0,0), P(1,1,0)) == L(P(0,0,0), P(1,1,0))
        assert L(P(0,0,0), P(1,1,0)) != L(P(0,0,0), P(2,1,0))
        print("Equality Test Passed")
    
    @staticmethod
    def parallel_test():
        L = Line3; P = Vector3
        def P2(x, y) -> Vector3:
            return Vector3(x, y, 0)

        assert L.AreParallel(
            L(P2(0,0), P2(1,0)),
            L(P2(0,2), P2(1,2))
        )
        assert L.AreParallel(
            L(P2(1,0), P2(0,0)),
            L(P2(0,2), P2(1,2))
        )
        assert L.AreParallel(
            L(P2(0,0), P2(0,1)),
            L(P2(1,0), P2(1,1))
        )
        assert L.AreParallel(
            L(P2(0,0), P2(1,0)),
            L(P2(1,2), P2(0,2))
        )
        assert L.AreParallel(
            L(P2(0,0), P2(1,1)),
            L(P2(-1,-1), P2(-3,-3))
        )
        assert not L.AreParallel(
            L(P2(0,0), P2(1,1)),
            L(P2(0,0), P2(-1,1))
        )
        print("Parallel Test Passed")

    @staticmethod
    def distance_to_point_test():
        L = Line3; P = Vector3
        l = L(P(0,0,0), P(1,0,0))
        assert l.get_distance_to_point(P(0,1,0)) == 1
        assert l.get_distance_to_point(P(-1,0,0)) == 0
        print("Distance To Point Test Passed")

    @staticmethod
    def line_shortest_distance_test(verbose: bool=False):
        L = Line3; P = Vector3
        thresh = 0.001

        def get_flags_str(l0, l1, thresh) -> str:
            return ', '.join([
                f"{label}: {func(l0, l1, thresh)}"
                for label, func in [
                    ('Parallel', L.AreParallel),
                    ('Intersecting', L.AreIntersecting),
                    ('Skew', L.AreSkew)
                ]
            ])

        l0 = L(P(0,0,0), P(1,0,0)); l1 = L(P(0,1,0), P(0,1,1))
        if verbose:
            print('Skew')
            print(get_flags_str(l0, l1, thresh))
            print(f"{L.ShortestDistance(l0, l1, thresh=thresh)=}")
            print(f"{L.Intersection(l0, l1, thresh=thresh)=}")
        assert L.AreSkew(l0,l1,thresh)
        assert L.ShortestDistance(l0, l1, thresh=thresh) == 1.0
        assert L.Intersection(l0, l1, thresh=thresh) is None

        if verbose:
            print('')

        l0 = L(P(0,0,0), P(1,0,0)); l1 = L(P(0,0,0), P(0,1,0)) + P(5,0,0)
        if verbose:
            print('Intersection')
            print(get_flags_str(l0, l1, thresh))
            print(f"{L.ShortestDistance(l0, l1, thresh=thresh)=}")
            print(f"{L.Intersection(l0, l1, thresh=thresh)=}")
        assert L.AreIntersecting(l0,l1,thresh)
        assert L.ShortestDistance(l0, l1, thresh=thresh) == 0
        assert L.Intersection(l0, l1, thresh=thresh) == Vector3(5,0,0)

        if verbose:
            print('')

        l0 = L(P(0,0,0), P(1,0,0)); l1 = L(P(0,0,0), P(1,0,0)) + P(0,5,0)
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
        Line3.equality_test()
        Line3.parallel_test()
        Line3.distance_to_point_test()
        Line3.line_shortest_distance_test(verbose=False)
    #endregion