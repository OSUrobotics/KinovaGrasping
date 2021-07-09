#!/usr/bin/env python3

#  Base class for all geometry in the simulation (the hand, the object, and the environment)
#    The assumption is that all geometry has at least
#       The raw geometry (the stl/obj/mesh) - in the form of vertex locations
#       One (or more) transforms
#           xml/urdf: the transform that is applied to the raw geometry as it is read in from the file
#              - this doesn't change
#           simulation matrix/location: The transform applied to the geometry by the simulation/IK solver, etc
#   In addition, define some useful debugging/calculation tools
#     A bounding box around the original mesh geometry
#     A signed distance function for the original mesh geometry

from numpy import array as nparray
from numpy import eye
from coordinate_system import CoordinateSystemTransform, BoundingBox
from bounding_box import BoundingBox
from signed_distance_fc import SignedDistanceFc
from data_directories_base import DataDirectoryBase


class GeometryBase(DataDirectoryBase):
    # If eg PyBullet or Mujoco or RViz is up and running
    sym_environment = None

    def __init__(self, cls_name, type_name, instance_name=""):
        """Created from an stl file, urdf, xml, etc
             use set functions to create geometry
             Actual geometry can be stored here or in the simulation (if running)"""
        super(DataDirectoryBase, self).__init__(cls_name, instance_name)
        self.name = instance_name
        # bounding box will be in self.mesh_to_world
        self.mesh_to_world = CoordinateSystemTransform((type_name, "origin"), (type_name, "world"))
        self.sdf = SignedDistanceFc()

        # If we're in the simulation, use this to get/set the geometry from the simulation
        self.sim_ref = None

    def __str__(self):
        """TODO """
        return "Geometry"

    def __repr__(self):
        return self.__str__()

    def get_vertices(self):
        """ Generator function that returns all vertices in the form of a 1x4 numpy array [x,y,z,1]
              Suitable for multiplying by a matrix
            Uses the sim_ref pointer """

        if not self.sim_ref:
            from numpy import random
            fake_data = random.rand(10, 3)
            for d in fake_data:
                yield nparray([d[0], d[1], d[2], 1])

    def get_from_bbox(self):
        """Just to make it easier to find
        @returns BoundingBox"""
        return self.mesh_to_world.bbox_from()

    def get_mesh_to_world_matrix(self):
        """Just to make it easier to find
        @returns nparray matrix"""
        return self.mesh_to_world.get_matrix()

    def get_world_to_mesh_matrix(self):
        """Just to make it easier to find
        @returns nparray matrix"""
        return self.mesh_to_world.get_inverse_matrix()

    def set_world_xform(self):
        """ Get the current world transform(s) from the simulator """
        raise NotImplementedError

    def set_signed_distance_fc(self):
        """ Once the geometry is set-up, call this"""
        raise NotImplementedError

    def render(self, draw_mesh_bbox=True):
        """TODO Draw geometry in Open GL
        @param draw_mesh_bbox Draw the mesh bounding box as well"""
        raise NotImplementedError


if __name__ == "__main__":

    my_obj = GeometryBase("test")
