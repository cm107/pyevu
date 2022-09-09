import os
from typing import Callable, Generic
from .solver import Solver, DType
from .. import Vector2, Vector3

class MassDamperSpring(Generic[DType], object):
    def __init__(
        self, m: float, c: float, k: float,
        center: DType=None, natural_length: float=None, direction: DType=None,
        name: str="Mass",
        record_meta: bool=True
    ):
        self.m = m
        self.c = c
        self.k = k
        self.center = center
        self.natural_length = natural_length
        self.direction = direction
        self.name = name
        self.record_meta = record_meta
        self.external_force_func: Callable[[float, Solver.State[DType]], float]=None

        # Need to be initialized seperately for get_generic_type to work correctly.
        self.integrator: Solver.LeapFrog[DType] = None
        self.meta: Solver.ForceStateMeta[DType] = None

    def initialize(self, init_state: Solver.State[DType]):
        gen_type = self.get_generic_type()
        self.integrator = Solver.LeapFrog[gen_type](init_state=init_state)
        self.meta = Solver.ForceStateMeta[gen_type]()

    def get_generic_type(self) -> type: # Note: Can't be called from __init__
        return self.__orig_class__.__args__[0]

    def is_valid_type(self) -> bool:
        gen_type = self.get_generic_type()
        if gen_type not in [float, Vector2, Vector3]:
            return False
        for val in [self.center, self.direction]:
            if val is not None and type(val) is not gen_type:
                return False
        return True

    @property
    def state(self) -> Solver.State[DType]:
        return self.integrator.state
    
    @property
    def rel_spring_p(self) -> DType:
        if self.direction is None:
            gen_type = self.get_generic_type()
            if gen_type is float:
                direction = 1
            elif gen_type is Vector2:
                direction = Vector2.right
            elif gen_type is Vector3:
                direction = Vector3.right
            else:
                raise TypeError
        else:
            direction = self.direction
        if self.center is None:
            gen_type = self.get_generic_type()
            if gen_type is float:
                center = 0
            elif gen_type is Vector2:
                center = Vector2.zero
            elif gen_type is Vector3:
                center = Vector3.zero
            else:
                raise TypeError
        else:
            center = self.center
        if self.natural_length is None:
            natural_length = 0
        else:
            natural_length = self.natural_length

        resting_p = center + natural_length * direction
        return self.state.p - resting_p

    def get_f(self, t: float=None) -> DType:
        f = 0
        if self.external_force_func is not None:
            f += self.external_force_func(t, self.state)
        f -= self.c * self.state.v
        f -= self.k * self.rel_spring_p
        return f
    
    def update_meta(self, t: float, f: DType):
        if self.record_meta:
            if t is None:
                raise Exception("Must provide t when recording meta.")
            state = self.state
            self.meta.step(Solver.ForceState(p=state.p, v=state.v, a=state.a, f=f, t=t))

    def update(self, dt: float=0.05, t: float=None):
        f = self.get_f(t=t)
        self.integrator.update(a=f / self.m, dt=dt)
        self.update_meta(t=t, f=f)

class MassDamperSpringSystem(Generic[DType], object):
    def __init__(
        self,
        objects: list[MassDamperSpring[DType]]
    ):
        self.objects = objects

    def get_generic_type(self) -> type: # Note: Can't be called from __init__
        return self.__orig_class__.__args__[0]

    def is_valid_type(self) -> bool:
        gen_type = self.get_generic_type()
        if gen_type not in [float, Vector2, Vector3]:
            return False
        for obj in self.objects:
            if obj.get_generic_type() is not gen_type:
                return False
        return True

    def run(
        self,
        tmax: float=10, dt: float=0.05,
        save_dir: str=None
    ):
        if not self.is_valid_type():
            raise TypeError
        t = 0
        
        for obj in self.objects:
            obj.update_meta(t=t, f=obj.get_f(t=t))

        while t < tmax:
            t += dt
            for obj in self.objects:
                obj.update(dt=dt, t=t)

        if any([obj.record_meta for obj in self.objects]):
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
            for obj in self.objects:
                obj_save_dir = None
                if save_dir is not None:
                    obj_save_dir = f"{save_dir}/{obj.name}"
                obj.meta.plot(save_dir=obj_save_dir)