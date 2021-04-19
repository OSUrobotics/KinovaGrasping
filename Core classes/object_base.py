#!/usr/bin/env python3

#  Given a simulation environment, can also grab the current matrix transform
#   for a model/mesh/object in the global coordinate system

from coordinate_system import CoordinateSystemTransformBase, BoundingBox
from scipy.spatial.transform import Rotation
from numpy import array as nparray

class ObjectBase():
    # If eg PyBullet or Mujoco or RViz is up and running
    sym_environment = None

    def __init__(self, name=""):
        """Created from an stl file, urdf, xml, etc
             Actual geometry can be stored here or in the simulation (if running)"""
        self.name = name

    def get_vertices(self):
        """ Generator function that returns all vertices in the form of a 1x4 numpy array [x,y,z,1]
              Suitable for multiplying by a matrix"""
        # Defined in derived class - this is dummy behavior which generates points in a 1x1x1 box
        from numpy import random
        fake_data = random.rand(10, 3)
        for d in fake_data:
            yield nparray([d[0], d[1], d[2], 1])

    def calc_bbox(self, orientation: Rotation = None) -> tuple:
        """Bounding box of mesh geometry in the current coordiante system
        @ returns bounding box as [ [lower left] [ upper right ] ]"""
        bbox_min = nparray([1e30, 1e30, 1e30])
        bbox_max = nparray([-1e30, -1e30, -1e30])

        # Note, under construction - need to set to input rotation or identity if none
        my_matrix = self.get_matrix()
        for v in self.get_vertices():
            v_transformed = my_matrix @ v
            bbox_min = min(v_transformed, bbox_min)
            bbox_max = max(v_transformed, bbox_max)
        return BoundingBox(tuple(bbox_min.tolist()), tuple(bbox_max.to_list()))

    def render(self):
        """Draw geometry in Open GL"""
        raise NotImplementedError


if __name__ == "__main__":

    my_obj = ObjectBase("test")
