from __future__ import annotations
import json
from typing import cast, List
from .transform import Transform
from .vector3 import Vector3
from .quat import Quat
from .timestamp import Timestamp

class GameObject:
    def __init__(self, name: str="GameObject", transform: Transform=None, children: List[GameObject]=None, timestamp: Timestamp=None):
        self.transform = transform if transform is not None else Transform(position=Vector3.zero, rotation=Quat.identity)
        self.transform.gameObject = self
        self.name = name
        self.timestamp = Timestamp() if timestamp is None else timestamp
        self.children = cast(List[GameObject], [])
        if children is not None:
            for child in children:
                child.transform.parent = self.transform

    # def _ApplyDiffToChildren(self, diffTransform: Transform):
    #     # Note: Be careful not to delete references to parent or gameObject.
    #     for go in self.children:
    #         newTransform = diffTransform * go.transform
    #         go.transform.position = newTransform.position
    #         go.transform.rotation = newTransform.rotation
    #         if len(go.children) > 0:
    #             # UpdateChildren(go.children, diffTransform)
    #             go._ApplyDiffToChildren(diffTransform)

    def _ChildrenUpdateWorldTransformationMatrix(self):
        for go in self.children:
            go.transform.UpdateWorldTransformationMatrix()
    

    def ToDict(self, utcList: List[float]=None, ensure_unique_timestamps: bool=True) -> dict:
        if ensure_unique_timestamps:
            if utcList is None:
                utcList = [self.timestamp.ToUtc()]
            else:
                utcList.append(self.timestamp.ToUtc())

        hierarchy = {
            'name': self.name,
            'timestamp': self.timestamp.ToUtc(),
            'transform': self.transform.ToDict(),
            'children': [child.ToDict(utcList=utcList, ensure_unique_timestamps=ensure_unique_timestamps) for child in self.children]
        }
        uniqueTimestamps = list(set(utcList))
        assert len(uniqueTimestamps) == len(utcList), "Found non-unique timestamps in hierarcy."
        return hierarchy
    
    @classmethod
    def FromDict(cls, hierarchy: dict) -> GameObject:
        return GameObject(
            transform=Transform.FromDict(hierarchy['transform']),
            name=hierarchy['name'],
            children=[GameObject.FromDict(childHierarcy) for childHierarcy in hierarchy['children']],
            timestamp=Timestamp.FromUtc(hierarchy['timestamp'])
        )
    
    def SaveHierarchy(self, path: str):
        json.dump(self.ToDict(), open(path, 'w'), ensure_ascii=False, indent=2)
    
    @classmethod
    def LoadHierarchy(cls, path: str) -> GameObject:
        hierarchy = json.load(open(path, 'r'))
        return GameObject.FromDict(hierarchy)