from __future__ import annotations
import os
import json
from typing import cast, List
from .transform import Transform
from .vector3 import Vector3
from .quaternion import Quaternion
from .timestamp import Timestamp

class GameObject:
    def __init__(self, transform: Transform=None, name: str="GameObject", children: List[GameObject]=None, timestamp: Timestamp=None):
        self.transform = transform if transform is not None else Transform(position=Vector3.zero, rotation=Quaternion.identity)
        self.transform.gameObject = self
        self.name = name
        self.timestamp = Timestamp() if timestamp is None else timestamp
        self.children = cast(List[GameObject], [])
        if children is not None:
            for child in children:
                child.transform.parent = self.transform

    def _ApplyDiffToChildren(self, diffTransform: Transform):
        # Note: Be careful not to delete references to parent or gameObject.
        for go in self.children:
            newTransform = diffTransform * go.transform
            go.transform.position = newTransform.position
            go.transform.rotation = newTransform.rotation
            if len(go.children) > 0:
                # UpdateChildren(go.children, diffTransform)
                go._ApplyDiffToChildren(diffTransform)
    

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
    
    @classmethod
    def UnitTest(cls):
        root = GameObject(name='root', transform=Transform.identity)
        
        # Burger
        burger = GameObject(name='burger', transform=Transform(Vector3.up * 0.5, Quaternion.identity))
        burger.transform.parent = root.transform
        
        # Upper Bun
        upperBun = GameObject(name='upperBun')
        upperBun.transform.parent = burger.transform
        upperBun.transform.localTransform = Transform.identity # Move to parent
        upperBun.transform.localPosition += Vector3.up * 0.25 # Move up a little
        upperBun.transform.localRotation *= Quaternion.AngleAxis(30, Vector3.up) # Rotate by 30 about y-axis

        # Lower Bun
        lowerBun = GameObject(name='lowerBun')
        lowerBun.transform.parent = burger.transform
        lowerBun.transform.localTransform = Transform.identity # Move to parent
        lowerBun.transform.localPosition += Vector3.down * 0.25 # Move down a little
        lowerBun.transform.localRotation *= Quaternion.AngleAxis(-50, Vector3.up) # Rotate by -50 about y-axis

        # Paddy
        paddy = GameObject(name='paddy')
        paddy.transform.parent = burger.transform
        paddy.transform.localTransform = Transform.identity # Move to parent
        paddy.transform.position = 0.5 * (upperBun.transform.position + lowerBun.transform.position) # Move paddy in between upperBun and lowerBun
        
        # Sesame Seeds
        paddyToUpperBunVec = upperBun.transform.position - paddy.transform.position # Radius of hemisphere
        sesameSeeds: List[GameObject] = []
        for i in range(10):
            seed = GameObject(name=f'seed{i}')
            seed.transform.parent = upperBun.transform
            seed.transform.position = paddy.transform.position
            rot = Quaternion.EulerVector(Vector3.RandomRange(x=(45, 135), y=(-180, 180), z=(0,0)), order='xyz')
            newTransform = seed.transform.Copy()
            newTransform *= Transform(paddyToUpperBunVec, rot)
            seed.transform.position = newTransform.position
            seed.transform.rotation = newTransform.rotation
            sesameSeeds.append(seed)

        print(root.ToDict())
        assert root.ToDict() == GameObject.FromDict(root.ToDict()).ToDict()
        savePath = '/tmp/hierarchyTest.json'
        root.SaveHierarchy(savePath)
        root0 = GameObject.LoadHierarchy(savePath)
        assert root.ToDict() == root0.ToDict()
        
        print("Before Removing Upper Bun")
        root.transform.PrintHierarchy()

        upperBun.transform.localRotation = Quaternion.Euler(180, 0, 0) * upperBun.transform.localRotation
        upperBun.transform.localPosition += Vector3.one * 5

        print("Before Removing Upper Bun")
        root.transform.PrintHierarchy()
