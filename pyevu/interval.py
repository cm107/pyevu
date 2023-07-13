from __future__ import annotations
from typing import Union

class Interval:
    def __init__(self, min: Union[float, int], max: Union[float, int]):
        self.min = min
        self.max = max
    
    def __str__(self) -> str:
        return f'Interval({self.min},{self.max})'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __key(self: V) -> tuple:
        return tuple([self.__class__] + list(self.__dict__.values()))

    def __hash__(self: V):
        return hash(self.__key())

    def __eq__(self: V, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented

    def __lt__(self, other: Interval) -> bool:
        if type(other) is Interval:
            return self.max <= other.min
        else:
            raise TypeError
    
    def __le__(self, other: Interval) -> bool:
        if type(other) is Interval:
            return self.max <= other.max
        else:
            raise TypeError

    def __gt__(self, other: Interval) -> bool:
        if type(other) is Interval:
            return self.min >= other.max
        else:
            raise TypeError
    
    def __ge__(self, other: Interval) -> bool:
        if type(other) is Interval:
            return self.min >= other.min
        else:
            raise TypeError

    def Contains(self, value: Union[float, int]) -> bool:
        return value >= self.min and value <= self.max
    
    @classmethod
    def Overlaps(cls, *args: Interval) -> bool:
        def overlaps(a: Interval, b: Interval) -> bool:
            return b.Contains(a.min) or b.Contains(a.max) \
                or a.Contains(b.min) or a.Contains(b.max)

        for i in range(len(args)):
            for j in range(i+1, len(args)):
                if not overlaps(args[i], args[j]):
                    return False
        return True

    @classmethod
    def Union(cls, *args: Interval) -> Interval:
        vals = [obj.min for obj in args] + [obj.max for obj in args]
        return Interval(min(vals), max(vals))
    
    @classmethod
    def Intersection(cls, *args: Interval) -> Union[Interval, None]:
        if (Interval.Overlaps(*args)):
            return Interval(
                min=max(*[obj.min for obj in args]),
                max=min(*[obj.max for obj in args])
            )
        else:
            return None

    @classmethod
    def Gap(cls, a: Interval, b: Interval) -> Interval:
        if (not Interval.Overlaps(a, b)):
            if (a < b):
                return Interval(min=a.max, max=b.min)
            elif (a > b):
                return Interval(min=b.max, max=a.min)
            else:
                raise Exception
        else:
            return None
    
    @property
    def length(self) -> float:
        return self.max - self.min
    
    @property
    def center(self) -> float:
        return (self.min + self.max) * 0.5

    @classmethod
    def Distance(cls, a: Interval, b: Interval) -> float:
        gap = Interval.Gap(a, b)
        return gap.length if gap is not None else 0
    
    def Clamp(self, val: float) -> float:
        if val < self.min:
            return self.min
        elif val > self.max:
            return self.max
        else:
            return val
    
    @staticmethod
    def IoU(i0: Interval, i1: Interval) -> float:
        """Intersection over Union (IoU)
        Same concept as BBox2D.IoU, but in 1D.
        """
        intersection = Interval.Intersection(i0, i1)
        if intersection is None:
            return 0
        else:
            overlap = intersection.length
            union = i0.length + i1.length - overlap
            return overlap / union

    @classmethod
    def split_at_intersection(cls, i0: Interval, i1: Interval) -> list[Interval]:
        intersection = cls.Intersection(i0, i1)
        if intersection is None or intersection.length == 0:
            # 2
            return [i0, i1]
        elif i0 == i1:
            # 1
            return [i0]
        else:
            # 3
            minVal = min(i0.min, i1.min)
            maxVal = max(i0.max, i1.max)
            return [
                cls(minVal, intersection.min),
                intersection,
                cls(intersection.max, maxVal)
            ]

    @staticmethod
    def unit_test_comparison_ops():
        assert Interval(-5,3) == Interval(-5,3)
        assert Interval(-5,3) != Interval(4,5)
        assert Interval(-5,3) < Interval(4,5)
        assert Interval(-5,3) <= Interval(4,5)
        assert Interval(4,5) > Interval(-5,3)
        assert Interval(4,5) >= Interval(-5,3)
        print("Passed Comparison Ops Unit Test")

    @staticmethod
    def unit_test_overlaps():
        intervals = [ # Intersects at Interval(1.5,2)
            Interval(-5,3),
            Interval(-3,2.5),
            Interval(1,2),
            Interval(1.1,2.1),
            Interval(1.5,3.5)
        ]
        assert Interval.Overlaps(*intervals)
        assert not Interval.Overlaps(*(intervals + [Interval(-2,-1)]))
        print("Passed Overlaps Unit Test")
    
    @staticmethod
    def unit_test_union():
        intervals = [ # Union at Interval(-5,3.5)
            Interval(-5,3),
            Interval(-3,2.5),
            Interval(1,2),
            Interval(1.1,2.1),
            Interval(1.5,3.5)
        ]
        assert Interval.Union(*intervals) == Interval(-5,3.5)
        print("Passed Union Unit Test")
    
    @staticmethod
    def unit_test_intersection():
        intervals = [ # Intersects at Interval(1.5,2)
            Interval(-5,3),
            Interval(-3,2.5),
            Interval(1,2),
            Interval(1.1,2.1),
            Interval(1.5,3.5)
        ]
        assert Interval.Intersection(*intervals) == Interval(1.5,2)
        print("Passed Intersection Unit Test")
    
    @staticmethod
    def unit_test_split_at_intersection():
        i0 = Interval(0,1); i1 = Interval(0.5,1.5)
        expectedResult = [
            Interval(0,0.5), Interval(0.5,1), Interval(1,1.5)
        ]
        assert Interval.split_at_intersection(i0,i1) == expectedResult
        assert Interval.split_at_intersection(i1,i0) == expectedResult
        assert Interval.split_at_intersection(i0,i0) == [i0]
        assert Interval.split_at_intersection(i1,i1) == [i1]

        i2 = Interval(10,11)
        assert Interval.split_at_intersection(i0,i2) == [i0,i2]
        assert Interval.split_at_intersection(i2,i0) == [i2,i0]
        assert Interval.split_at_intersection(i1,i2) == [i1,i2]
        assert Interval.split_at_intersection(i2,i1) == [i2,i1]

        i3 = Interval(1,2)
        assert Interval.split_at_intersection(i0,i3) == [i0,i3]
        assert Interval.split_at_intersection(i3,i0) == [i3,i0]
        print("Passed Split At Intersection Test")

    @staticmethod
    def unit_test():
        Interval.unit_test_comparison_ops()
        Interval.unit_test_overlaps()
        Interval.unit_test_union()
        Interval.unit_test_intersection()
        Interval.unit_test_split_at_intersection()
        print("Passed All Unit Tests for Interval Class")