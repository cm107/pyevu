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
    def Overlaps(cls, a: Interval, b: Interval) -> bool:
        return b.Contains(a.min) or b.Contains(a.max) \
            or a.Contains(b.min) or a.Contains(b.max)

    @classmethod
    def Union(cls, a: Interval, b: Interval) -> Interval:
        return Interval(
            min=a.min if a.min <= b.min else b.min,
            max=a.max if a.max >= b.max else b.max
        )
    
    @classmethod
    def Intersection(cls, a: Interval, b: Interval) -> Union[Interval, None]:
        if (Interval.Overlaps(a, b)):
            return Interval(min=max(a.min, b.min), max=min(a.max, b.max))
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
