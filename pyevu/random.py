import random

class Random:
    @classmethod
    def FloatRange(cls, min: float, max: float) -> float:
        delta = max - min
        return min + delta * random.random()
    
    @classmethod
    def IntRange(cls, min: int, max: int) -> int:
        return random.randrange(min, max+1) # +1 to enclude end of interval