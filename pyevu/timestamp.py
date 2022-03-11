from __future__ import annotations
from typing import Union
from datetime import datetime, timedelta

class Timestamp:
    def __init__(self, t: datetime=None):
        self.t = t if t is not None else datetime.now()

    def __str__(self) -> str:
        return str(self.ToUtc())
    
    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: Timestamp) -> bool:
        if type(other) is Timestamp:
            return self.ToUtc() == other.ToUtc()
        else:
            return False

    def __sub__(self, other: Union[Timestamp, datetime]) -> timedelta:
        if type(other) is Timestamp:
            return self.t - other.t
        elif type(other) is datetime:
            return self.t - other
        else:
            raise TypeError(f"Can't subtract {type(other).__name__} from {type(self).__name__}")

    def __rsub__(self, other: Union[Timestamp, datetime]) -> timedelta:
        if type(other) is Timestamp:
            return other.t - self.t
        elif type(other) is datetime:
            return other - self.t
        else:
            raise TypeError(f"Can't subtract {type(self).__name__} from {type(other).__name__}")

    def ToUtc(self) -> float:
        return (self.t - datetime(1970,1,1)).total_seconds()
    
    @classmethod
    def FromUtc(cls, utc: float) -> Timestamp:
        return Timestamp(datetime.utcfromtimestamp(utc))
    
    def Copy(self) -> Timestamp:
        return Timestamp.FromUtc(self.ToUtc())