from __future__ import annotations
from typing import Callable, Generic, TypeVar


C = TypeVar('C') # C for Callable

class Event(Generic[C]):
    def __init__(self):
        self._hooks: list[Callable[[C], None]] = []
    
    def __iter__(self) -> list[Callable[[C], None]]:
        return iter(self._hooks)

    def __eq__(self, other) -> bool:
        if callable(other):
            return other in self._hooks
        return NotImplemented

    def __add__(self, other) -> Event[C]:
        if callable(other):
            self._hooks.append(other)
            return self
        else:
            raise TypeError(f"{type(other).__name__} is not callable.")

    def __sub__(self, other) -> Event[C]:
        if callable(other):
            self._hooks.remove(other)
            return self
        else:
            raise TypeError(f"{type(other).__name__} is not callable.")

    def Subscribe(self, other: Callable[[C], None]):
        self += other
    
    def Unsubscribe(self, other: Callable[[C], None]):
        self -= other
    
    def IsSubscribed(self, other: Callable[[C], None]) -> bool:
        return other in self

    def Invoke(self, input: C=None):
        if input is not None:
            for hook in self._hooks:
                hook(input)
        else:
            for hook in self._hooks:
                hook()

    @staticmethod
    def debug():
        event0: Event[None] = Event[None]()

        def funcA():
            print("This is Func-Aye!")
        
        def funcB():
            print("This is Func-Bay!")

        event0 += funcA
        event0 += funcB
        event0 -= funcA
        event0 += funcA
        event0 += funcA
        if funcA not in event0:
            event0 += funcA

        event0.Invoke()

        print("Remove all Aye!")

        while funcA in event0:
            event0 -= funcA
        
        event0.Invoke()
