from __future__ import annotations
import math
from typing import Callable
import matplotlib.pyplot as plt
import os

class State:
    def __init__(self, p: float=None, v: float=None, a: float=None):
        self.p = p
        self.v = v
        self.a = a
    
    @classmethod
    @property
    def zero(self) -> State:
        return State(p=0, v=0, a=0)

class LeapFrog:
    def __init__(self, init_state: State):
        self.state = init_state
    
    def next_state(self, a: float, dt: float=0.05) -> State:
        half_v = self.state.v + self.state.a * dt * 0.5
        p = self.state.p + half_v * dt
        v = half_v + a * dt / 2
        return State(p=p, v=v, a=a)
    
    def update(self, a: float, dt: float=0.05):
        self.state = self.next_state(a=a, dt=dt)
    
    @property
    def p(self) -> float:
        return self.state.p
    
    @property
    def v(self) -> float:
        return self.state.v
    
    @property
    def a(self) -> float:
        return self.state.a
    
    @a.setter
    def a(self, value: float):
        self.state.a = value

class MassDamperSpring:
    def __init__(
        self, m: float, c: float, k: float,
        init_state: State=State.zero,
        center: float=0, natural_length: float=0,
        name: str="Mass",
        record_meta: bool=True
    ):
        self.m = m
        self.c = c
        self.k = k
        self.integrator: LeapFrog = LeapFrog(init_state=init_state)
        self.center = center
        self.natural_length = natural_length
        self.name = name
        self.meta = MassDamperSpring.Meta()
        self.record_meta = record_meta
        self.external_force_func: Callable[[float, State], float]=None

    @property
    def state(self) -> State:
        return self.integrator.state
    
    @property
    def rel_spring_p(self) -> float:
        resting_p = self.center + self.natural_length
        return self.state.p - resting_p

    def get_f(self, t: float=None) -> float:
        f = 0
        if self.external_force_func is not None:
            f += self.external_force_func(t, self.state)
        f -= self.c * self.state.v
        f -= self.k * self.rel_spring_p
        return f
    
    def update_meta(self, t: float, f: float):
        if self.record_meta:
            if t is None:
                raise Exception("Must provide t when recording meta.")
            self.meta.step(t=t, f=f, state=self.state)

    def update(self, dt: float=0.05, t: float=None):
        f = self.get_f(t=t)
        self.integrator.update(a=f / self.m, dt=dt)
        self.update_meta(t=t, f=f)

    class Meta:
        def __init__(self):
            self.tlist: list[float] = []; self.flist: list[float] = []
            self.state_list: list[State] = []
        
        def get_pvalist(self) -> tuple[list[float], list[float], list[float]]:
            plist: list[float] = []
            vlist: list[float] = []
            alist: list[float] = []
            for state in self.state_list:
                plist.append(state.p)
                vlist.append(state.v)
                alist.append(state.a)
            return plist, vlist, alist

        def step(
            self,
            t: float, f: float, state: State
        ):
            self.tlist.append(t)
            self.flist.append(f)
            self.state_list.append(state)

        def _plot(
            self,
            tlist: list[float], flist: list[float],
            target_list: list[float], target_label: str,
            save_path: str=None
        ):
            ax = plt.subplot(111)
            for i in range(len(tlist)):
                t = tlist[i]; f = flist[i]; target = target_list[i]
                # plt.plot(
                #     t, f, marker='+',
                #     color='red'
                # )
                plt.plot(
                    t, target, marker='+',
                    color='blue'
                )

            plt.title("Force Impulse Simulation")
            plt.xlabel("time (in seconds)")
            plt.ylabel(target_label)

            # Re-position Legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])    
            legend = plt.legend(['Applied Force', target_label], loc='upper left', bbox_to_anchor=(1, 1))
            legend.legendHandles[0].set_color('red')
            legend.legendHandles[1].set_color('blue')

            if save_path is not None:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.clf()
            plt.close('all')
        
        def plot(self, save_dir: str=None):
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)

            plist, vlist, alist = self.get_pvalist()
            for target_list, target_label in [
                (plist, "Position (in m)"),
                (vlist, "Velocity (in m/s)"),
                (alist, "Acceleration (in m/s^2)")
            ]:
                save_path = None
                if save_dir is not None:
                    basename = target_label.split(" ")[0].lower()
                    save_path = f"{save_dir}/{basename}.png"
                self._plot(
                    tlist=self.tlist, flist=self.flist,
                    target_list=target_list,
                    target_label=target_label,
                    save_path=save_path
                )

class MassDamperSpringSystem:
    def __init__(
        self,
        objects: list[MassDamperSpring]
    ):
        self.objects = objects

    def run(
        self,
        tmax: float=10, dt: float=0.05,
        save_dir: str=None
    ):
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

def sine_impulse_force(t: float, f: float, w: float, td: float, t0: float=0) -> float:
    t -= t0
    if t < 0 or t > td:
        return 0
    else:
        return f * math.sin(w * t)

def force_func(t: float, state: State) -> float:
    f = 0
    td = 1; w = math.pi / td; t0 = 0; fi = 1; f += sine_impulse_force(t=t, f=fi, w=w, td=td, t0=t0)
    td = td / 2; w = math.pi / td; t0 = 5; fi = fi / 2; f += sine_impulse_force(t=t, f=fi, w=w, td=td, t0=t0)
    print(f"{t=}, {f=}")
    return f


if False: # Overdamped
    obj1 = MassDamperSpring(
        m=10, c=2*(10*100*4)**0.5, k=100,
        init_state=State(p=1, v=0, a=0),
        center=0, natural_length=1,
        name="m1",
        record_meta=True
    )
    obj2 = MassDamperSpring(
        m=10*2, c=2*(10*2*100*4)**0.5, k=100,
        init_state=State(p=2, v=0, a=0),
        center=1, natural_length=1,
        name="m2",
        record_meta=True
    )
else: # Critically Damped
    obj1 = MassDamperSpring(
        m=10, c=(10*100*4)**0.5, k=100,
        init_state=State(p=1, v=0, a=0),
        center=0, natural_length=1,
        name="m1",
        record_meta=True
    )
    obj2 = MassDamperSpring(
        m=10*2, c=(10*2*100*4)**0.5, k=100,
        init_state=State(p=2, v=0, a=0),
        center=1, natural_length=1,
        name="m2",
        record_meta=True
    )

def obj1_external_force_func(t: float, state: State) -> float:
    return obj2.m * obj2.state.a

obj1.external_force_func = obj1_external_force_func

def obj2_external_force_func(t: float, state: State) -> float:
    obj2.center = obj1.state.p
    return -obj1.m * obj1.state.a + force_func(t, state)

obj2.external_force_func = obj2_external_force_func

MassDamperSpringSystem([obj1, obj2]).run(
    tmax=10, dt=0.05,
    save_dir="dump"
)