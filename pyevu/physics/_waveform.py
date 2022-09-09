import math
from .solver import DType

class Waveform:
    @staticmethod
    def sine_impulse(t: float, a: DType, w: float, td: float, t0: float=0) -> DType:
        t -= t0
        if t < 0 or t > td:
            return 0
        else:
            return a * math.sin(w * t)