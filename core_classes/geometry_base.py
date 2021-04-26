#!/usr/bin/env python3

#  Given a simulation environment, can also grab the current matrix transform
#   for a model/mesh/object in the global coordinate system
#  Coordinate system transforms all objects should have
#   from_mesh: The transform that is in the xml/urdf file etc
#   to_world: The transform that positions the object in the world (after applying from_mesh

from numpy import array as nparray
from numpy import eye
from coordinate_system import CoordinateSystemTransformBase, BoundingBox
from bounding_box import BoundingBox
from signed_distance_fc import SignedDistanceFc
from data_directories_base import DataDirectoryBase

class GeometryBase(DataDirectoryBase):
    # If eg PyBullet or Mujoco or RViz is up and running
    sym_environment = None

    def __init__(self, name=""):
        """Created from an stl file, urdf, xml, etc
             use set functions to create geometry
             Actual geometry can be stored here or in the simulation (if running)"""
        self.name = name
        self.mesh_bbox = None
        self.sdf = SignedDistanceFc()

    def get_vertices(self):
        """ Generator function that returns all vertices in the form of a 1x4 numpy array [x,y,z,1]
              Suitable for multiplying by a matrix"""
        # Defined in derived class - this is dummy behavior which generates points in a 1x1x1 box
        from numpy import random
        fake_data = random.rand(10, 3)
        for d in fake_data:
            yield nparray([d[0], d[1], d[2], 1])

    def get_mesh_to_world(self):
        """ Transformation that takes the mesh to the world as a series of CoordinateSystemTransforms
        transforms should come out: [ mesh to something, something to something 2, something 2 to world ]
        Over-ride in derived classes
        @return CoordinteSystemTransformBase"""
        # Default is identity transform from mesh to world
        return [CoordinateSystemTransformBase("Mesh", "World")]

    def get_mesh_to_world_matrix(self):
        """Multiply the above matrices together
        @return numpy matrix"""
        m = eye(4)
        for c in self.get_mesh_to_world():
            m = c.get_matrix @ m

    def render(self, draw_mesh_bbox=True):
        """Draw geometry in Open GL
        @param draw_mesh_bbox Draw the mesh bounding box as well"""
        raise NotImplementedError


if __name__ == "__main__":

    my_obj = GeometryBase("test")
