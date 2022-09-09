import math
from pyevu import Vector3
from pyevu.physics import MassDamperSpring, MassDamperSpringSystem, Waveform, Solver

def force_func(t: float, state: Solver.State[Vector3]) -> Vector3:
    f = Vector3.zero
    td = 1; w = math.pi / td; t0 = 0; fd = Vector3(1,2,3).normalized; fi = 1 * fd; f += Waveform.sine_impulse(t=t, a=fi, w=w, td=td, t0=t0)
    td = td / 2; w = math.pi / td; t0 = 5; fd = Vector3(3,2,1).normalized; fi = 0.5 * fd; f += Waveform.sine_impulse(t=t, a=fi, w=w, td=td, t0=t0)
    print(f"{t=}, {f=}")
    return f

if False: # Overdamped
    m1=10; c1=2*(10*100*4)**0.5; k1=100,
    m2=10*2; c2=2*(10*2*100*4)**0.5; k2=100
else: # Critically Damped
    m1=10; c1=(10*100*4)**0.5; k1=100
    m2=10*2; c2=(10*2*100*4)**0.5; k2=100

obj1 = MassDamperSpring[Vector3](
    m=m1, c=c1, k=k1,
    center=Vector3.zero, natural_length=1, direction=Vector3.right,
    name="m1",
    record_meta=True
)
obj1.initialize(Solver.State(p=Vector3.right, v=Vector3.zero, a=Vector3.zero))
obj2 = MassDamperSpring[Vector3](
    m=m2, c=c2, k=k2,
    center=Vector3.zero, natural_length=1, direction=Vector3.right,
    name="m2",
    record_meta=True
)
obj2.initialize(Solver.State(p=Vector3.right * 2, v=Vector3.zero, a=Vector3.zero))

def obj1_external_force_func(t: float, state: Solver.State[Vector3]) -> Vector3:
    return obj2.m * obj2.state.a

obj1.external_force_func = obj1_external_force_func

def obj2_external_force_func(t: float, state: Solver.State[Vector3]) -> Vector3:
    obj2.center = obj1.state.p
    return -obj1.m * obj1.state.a + force_func(t, state)

obj2.external_force_func = obj2_external_force_func

MassDamperSpringSystem[Vector3]([obj1, obj2]).run(
    tmax=10, dt=0.05,
    save_dir="dump"
)