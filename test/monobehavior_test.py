from copy import copy
from typing import Callable, Generic, TypeVar
import uuid
from pyevu import Event

class MonoBehavior:
    def __init__(self):
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        if not self._enabled and value:
            self._enabled = value
            self.OnEnable()
        elif self._enabled and not value:
            self._enabled = value
            self.OnDisable()

    def OnEnable(self):
        """Override"""
        pass

    def OnDisable(self):
        """Override"""
        pass

    def _Awake(self):
        if self.enabled:
            self.OnEnable()
            self.Awake()

    def Awake(self):
        """Override"""
        pass
    
    def _Update(self):
        if self.enabled:
            self.Update()

    def Update(self):
        """Override"""
        pass

T = TypeVar('T')

class BaseHandler(Generic[T]):
    def __init__(self, objects: list[T]=None):
        self._objects = objects if objects is not None else []
        self.id = uuid.uuid4()
    
    def __str__(self) -> str:
        return f"[{', '.join([obj.__str__() for obj in self])}]"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __key(self) -> tuple:
        return tuple([self.__class__, self.id])

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self) -> T:
        if self.n < len(self):
            result = self._objects[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration
    
    def __getitem__(self, idx):
        if type(idx) is int:
            return self._objects[idx]
        elif type(idx) is slice:
            return self._objects[idx.start:idx.stop:idx.step]
        else:
            raise TypeError
    
    def __setitem__(self, idx, value):
        if type(idx) is int:
            self._objects[idx] = value
        elif type(idx) is slice:
            self._objects[idx.start:idx.stop:idx.step] = value
        else:
            raise TypeError

    def copy(self):
        """Shallow copy. Keep references."""
        return type(self)(copy.copy(self._objects))

    def deepcopy(self):
        """Deep copy. Copy to new location in memory."""
        return type(self)(copy.deepcopy(self._objects))

    def get(self, func: Callable[[T], bool]=None, **kwargs) -> T | None:
        if func is not None:
            for obj in self:
                if func(obj):
                    return obj
        else:
            for obj in self:
                is_match = True
                for key, val in kwargs.items():
                    if getattr(obj, key) != val:
                        is_match = False
                        break
                if is_match:
                    return obj
        return None
    
    def search(self, func: Callable[[T], bool]=None, **kwargs):
        objects: list[T] = []
        if func is not None:
            for obj in self:
                if func(obj):
                    objects.append(obj)
        else:
            for obj in self:
                is_match = True
                for key, val in kwargs.items():
                    if getattr(obj, key) != val:
                        is_match = False
                        break
                if is_match:
                    objects.append(obj)
        return type(self)(objects)
    
    def to_dict_list(self) -> list[dict]:
        return [
            (
                obj.to_dict() if hasattr(obj, "to_dict")
                else obj.__dict__
            )
            for obj in self
        ]
    
    def append(self, obj: T):
        self._objects.append(obj)
    
    def remove(self, obj: T):
        self._objects.remove(obj)

    def pop(self, idx: int=None) -> T:
        if idx is None:
            idx = len(self._objects) - 1
        return self._objects.pop(idx)

    def index(self, i: T | Callable[[T], bool]=None, **kwargs) -> int:
        if len(self) == 0:
            raise IndexError(f"{type(self).__name__} is empty.")
        elif i is not None:
            if type(i) is type(self[0]):
                return self._objects.index(i)
            elif callable(i):
                for idx, obj in enumerate(self):
                    if i(obj):
                        return idx
                raise ValueError
            else:
                raise TypeError
        elif len(kwargs) > 0:
            for idx, obj in enumerate(self):
                is_match = True
                for key, val in kwargs.items():
                    if getattr(obj, key) != val:
                        is_match = False
                        break
                if is_match:
                    return idx
        else:
            raise ValueError("Must provide parameters.")

B = TypeVar("B", bound=MonoBehavior) # B can only be a MonoBehavior or subtype of MonoBehavior

class BehaviorHandler(BaseHandler[B]):
    def __init__(self, behaviors: list[B]=None):
        super().__init__(behaviors)
        self.OnAwake = Event[None]()
        self.OnUpdate = Event[None]()

    def _Subscribe(self, behavior: B):
        self.OnAwake += behavior._Awake
        self.OnUpdate += behavior._Update

    def _Unsubscribe(self, behavior: B):
        self.OnAwake -= behavior._Awake
        self.OnUpdate -= behavior._Update

    def append(self, obj: B):
        super().append(obj)
        self._Subscribe(obj)        
    
    def remove(self, obj: B):
        self._objects.remove(obj)
        self._Unsubscribe(obj)

    def pop(self, idx: int=None) -> T:
        obj = self._objects.pop(idx)
        self._Unsubscribe(obj)
        return obj

raise NotImplementedError