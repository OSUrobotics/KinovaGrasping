#!/usr/bin/env python3

# The base class for all of the hand mesh classes
# Assumptions:
#   URDF with STLs for the geometry of the hand (palm, fingers, etc)
#   The geometry in the file(s) is in the hand open position
#   Possible xml transformation or other transform to scale it, orient it before attaching it to the wrist
#   This class does NOT move the actual geometry - use the simulator or other method to do that
# Transforms:
#   A transform to move it from the raw mesh location to the location it's expected to be in before applying
#     the transform to put it at the end effector
#   A transform to scale it/place it at the end of the end effector

from geometry_base import GeometryBase
from coordinate_system import CoordinateSystemBase, CoordinateSystemTransformBase


class HandGeometry(GeometryBase):
    # What pose the hand is in before starting to close fingers
    pose_preshape_type = {"cylindrical", "spherical", "oneside", "spread ang"}

    def __init__(self, hand_name):
        """ pass the class and instance name on
        @param cls_name: derived class
        @param instance_name: Optional string for name of object"""
        super(GeometryBase, self).__init__(cls_name=HandGeometry, type_name="hand", instance_name=hand_name)

        # Will be either vertices that are read in or a pointer to the mesh object in the simulator
        self.mesh = None
        self.preshape = "cylindrical"
        self.handspan_coords = CoordinateSystemBase(("hand", "handspan"))
        self.wrist_coords = CoordinateSystemBase(("hand", "wrist"))
        self.hand_wrist = CoordinateSystemTransformBase(("hand", "wrist"))
        self.hand_handspan = CoordinateSystemTransformBase(("hand", "handspan"))
        self.xml_urdf = CoordinateSystemTransformBase(("hand", "xmlOrURDF"))

    def read_hand_from_file(self):
        """ Read in the stl file for the corresponding base shape
        Sets all but handspan data"""
        raise NotImplementedError

    def read_handspan_from_file(self):
        """ Read in hand span info"""
        raise NotImplementedError


if __name__ == "__main__":

    my_obj = HandGeometry("test")
