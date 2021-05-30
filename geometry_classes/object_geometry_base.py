#!/usr/bin/env python3

# The base class for all of the object meshes (things we pick up) classes
# Assumptions:
#   STL, xml file, or mesh generation for the object
#   Possible xml transformation or other transform to scale it, orient it so the base is down and at 0,0,0 (object
#    coordinate system) and
#   A transform to scale it/place it in the environment

from geometry_base import GeometryBase
from coordinate_system import CoordinateSystemTransformBase


class ObjectGeometryBase(GeometryBase):

    def __init__(self, cls_name, instance_name):
        """ pass the class and instance name on
        @param cls_name: derived class
        @param instance_name: Optional string for name of object"""
        super(GeometryBase, self).__init__(cls_name=cls_name, type_name="object", instance_name=instance_name)

        # Will be either vertices that are read in or a pointer to the mesh object in the simulator
        self.mesh = None
        self.mesh_to_base = CoordinateSystemTransformBase(("object", "mesh"), ("object", "base"))
        self.mesh_to_xml = CoordinateSystemTransformBase(("object", "mesh"), ("object", "xmlOrURDF"))

    def get_vertices(self):
        """ Generator function that returns all vertices in the form of a 1x4 numpy array [x,y,z,1]
              Suitable for multiplying by a matrix"""
        # Assumes either a mesh or the sim pointer are defined
        if self.mesh:
            for v in self.mesh.vertices:
                yield v
        else:
            yield super(GeometryBase, self).get_vertices()


class ObjectCanonical(ObjectGeometryBase):
    """ These are the cylinders/cones/cubes"""
    orientations = {"original", "sideways"}
    def __init__(self, shape_name):
        """ Which object to create, and an optional scale/orient re-position
        @param shape_name: one of DerivedDataBase.defined_object_names
        @param instance_name: Optional string for name of object"""
        super(ObjectGeometryBase, self).__init__(ObjectCanonical, shape_name)
        if shape_name not in DataDirectoryBase.defined_object_names:
            raise KeyError("Object name: {0} not found in defined object names".format(shape_name))

        self.mesh = self.read_shape_from_file()

    def read_shape_from_file(self):
        """ Read in the stl file for the corresponding base shape"""
        raise NotImplementedError

    def set_orientation(self, orientation="original", rotate_around_y = 0):
        """Set the coordinate system transform(s) so that the object is oriented"""
        raise NotImplementedError

    def set_scale_to_hand(self, hand: HandGeometryBase, scale_per = (1, 1, 1)):
        """ TODO Assumes orientation already set
         Orient the object and scale it to the hand size"""
        raise NotImplementedError


if __name__ == "__main__":

    my_obj = GeometryBase("test")
