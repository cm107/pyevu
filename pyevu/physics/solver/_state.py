from __future__ import annotations
from typing import Type, Generic, TYPE_CHECKING, Union
from ... import Vector2, Vector3
from . import DType

class State(Generic[DType], object):
    def __init__(self, p: DType=None, v: DType=None, a: DType=None):
        self.p = p
        self.v = v
        self.a = a

    def get_generic_type(self) -> type: # Note: Can't be called from __init__
        return self.__orig_class__.__args__[0]

    def is_valid_type(self) -> bool:
        gen_type = self.get_generic_type()
        if gen_type not in [float, Vector2, Vector3]:
            return False
        for val in [self.p, self.v, self.a]:
            if type(val) is not gen_type:
                return False
        return True

    @staticmethod
    def test_is_valid_type():
        assert State[Vector3](p=Vector3.zero, v=Vector3.zero, a=Vector3.zero).is_valid_type()
        assert not State[Vector3](p=Vector3.zero, v=Vector2.zero, a=Vector3.zero).is_valid_type()
        assert not State[Vector2](p=Vector3.zero, v=Vector3.zero, a=Vector3.zero).is_valid_type()
        print(f"Passed: {State.__name__} is_valid_type is working correctly")

    @staticmethod
    def test_bench():
        State.test_is_valid_type()

    # @classmethod
    # @property
    # def zero(self: DType) -> State0[DType]: # This won't work.
    #     # gen_type = State0[DType]().get_generic_type()
    #     gen_type = self.get_generic_type(self)
    #     print(f"{gen_type=}")
    #     if gen_type is float:
    #         return State0[float](p=0, v=0, a=0)
    #     else:
    #         raise TypeError

    @staticmethod
    def zero_state(gen_type: Union[Type[float], type[Vector2], type[Vector3]]) -> Union[float, Vector2, Vector3]:
        if gen_type is float:
            return 0
        elif gen_type is Vector2:
            return Vector2.zero
        elif gen_type is Vector3:
            return Vector3.zero
        else:
            raise TypeError

class ForceState(Generic[DType], State[DType]):
    def __init__(self, p: DType, v: DType, a: DType, f: DType, t: float):
        super().__init__(p=p, v=v, a=a)
        self.f = f
        self.t = t
    
    @property
    def base(self) -> State[DType]:
        return State[DType](p=self.p, v=self.v, a=self.a)
    
    def is_valid_type(self) -> bool:
        gen_type = self.get_generic_type()
        if gen_type not in [float, Vector2, Vector3]:
            return False
        for val in [self.p, self.v, self.a, self.f]:
            if type(val) is not gen_type:
                return False
        if type(self.t) is not float:
            return False
        return True

class ForceStateMeta(Generic[DType], object):
    def __init__(self):
        self.data: list[ForceState[DType]] = []
    
    def get_generic_type(self) -> type: # Note: Can't be called from __init__
        return self.__orig_class__.__args__[0]

    def is_valid_type(self) -> bool:
        gen_type = self.get_generic_type()
        if gen_type not in [float, Vector2, Vector3]:
            return False
        return True

    def get_pvaft_list(self) -> tuple[list[DType], list[DType], list[DType], list[DType], list[float]]:
        plist: list[DType] = []; vlist: list[DType] = []; alist: list[DType] = []
        flist: list[DType] = []; tlist: list[float] = []
        for state in self.data:
            plist.append(state.p)
            vlist.append(state.v)
            alist.append(state.a)
            flist.append(state.f)
            tlist.append(state.t)
        return plist, vlist, alist, flist, tlist

    def step(self, state: ForceState[DType]):
        self.data.append(state)
    
    from ._force_state_meta_plot import plot
