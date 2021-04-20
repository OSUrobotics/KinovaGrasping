#!/usr/bin/env python3

# The base class for all of the object (things we pick up) classes
# Assumptions:
#   STL or xml file for the object
#   Possible xml transformation or other transform to scale it, orient it so the base is down and at 0,0,0 (object
#    coordinate system) and
#   A transform to scale it/place it in the environment

from object_base import ObjectBase


class ObjectObjectBase(ObjectBase):
    def __init__(self):
