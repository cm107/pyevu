import math

deg2rad = math.pi / 180
rad2deg = 180 / math.pi

def lerp(start: float, end: float, interpolationRatio: float) -> float:
    return start + (end - start) * interpolationRatio 