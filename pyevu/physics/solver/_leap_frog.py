from typing import Generic, TYPE_CHECKING
from ... import Vector2, Vector3
from ._state import State
from . import DType

class LeapFrog(Generic[DType], object):
    def __init__(self, init_state: State[DType]):
        self.state = init_state
    
    def get_generic_type(self) -> type: # Note: Can't be called from __init__
        return self.__orig_class__.__args__[0]
    
    def is_valid_type(self) -> bool:
        gen_type = self.get_generic_type()
        if gen_type not in [float, Vector2, Vector3]:
            return False
        if gen_type is not self.state.get_generic_type():
            return False
        if not self.state.is_valid_type():
            return False
        return True

    def next_state(self, a: DType, dt: float=0.05) -> State[DType]:
        half_v = self.state.v + self.state.a * dt * 0.5
        p = self.state.p + half_v * dt
        v = half_v + a * dt / 2
        return State[DType](p=p, v=v, a=a)
    
    def update(self, a: DType, dt: float=0.05):
        self.state = self.next_state(a=a, dt=dt)
    
    @property
    def p(self) -> DType:
        return self.state.p
    
    @property
    def v(self) -> DType:
        return self.state.v
    
    @property
    def a(self) -> DType:
        return self.state.a
    
    @a.setter
    def a(self, value: DType):
        self.state.a = value

    @staticmethod
    def test_is_valid_type():
        assert LeapFrog[Vector3](init_state=State[Vector3](p=Vector3.zero, v=Vector3.zero, a=Vector3.zero)).is_valid_type()
        assert not LeapFrog[Vector3](init_state=State[Vector3](p=Vector3.zero, v=Vector2.zero, a=Vector3.zero)).is_valid_type()
        assert not LeapFrog[Vector2](init_state=State[Vector3](p=Vector3.zero, v=Vector3.zero, a=Vector3.zero)).is_valid_type()
        assert not LeapFrog[Vector3](init_state=State[Vector2](p=Vector3.zero, v=Vector3.zero, a=Vector3.zero)).is_valid_type()
        print(f"Passed: {LeapFrog.__name__} is_valid_type is working correctly")

    @staticmethod
    def test_bench():
        LeapFrog.test_is_valid_type()
