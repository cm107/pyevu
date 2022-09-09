from typing import TypeVar
from ... import Vector2, Vector3

DType = TypeVar('DType', float, Vector2, Vector3)

class Solver:
    from ._state import State, ForceState, ForceStateMeta
    from ._leap_frog import LeapFrog
