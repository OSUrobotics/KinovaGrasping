#!/usr/bin/env python3

# The base class for all of the object meshes (things we pick up) classes
# Assumptions:
#   STL or xml file for the object
#   Possible xml transformation or other transform to scale it, orient it so the base is down and at 0,0,0 (object
#    coordinate system) and
#   A transform to scale it/place it in the environment

from geometry_base import GeometryBase
from coordinate_system import CoordinateSystemTransformBase


class ObjectObjectBase(GeometryBase):
    def __init__(self):
        self.mesh = None
        self.object_coord = CoordinateSystemTransformBase("Mesh", "ObjectBase")
