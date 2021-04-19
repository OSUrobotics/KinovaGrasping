#!/usr/bin/env python3

# Signed distance function implementation
# Given a point in space, transform it into the SDF space and find the SDF and/or its gradient
# SDF may be cached. The filename will depend on the object and the grid size (computed automatically)

from coordinate_system import CoordinateSystemTransformBase, BoundingBox
from object_base import ObjectBase

class SignedDistanceFc:
    def __init__(self):
        """Initialization will depend on mesh type"""
        self.bbox = BoundingBox()

        # Some of these might be defined in whatever library we end up using
        self.sampling = (0.1, 0.1, 0.1)   # Size of cube
        self.grid_size = (10, 10, 10)     # Number of grid cells in each direction
        self.padding = 0.1                # percentage empty space around object

        # These are for transforming to/from the object
        self.from_mesh_to_sdf = CoordinateSystemTransformBase("mesh", "signed_distance_function")

        self.obj_mesh = None

    def __str__(self):
        """ Print out stats about self"""
        return "SDF: {0} grid size {1}".format(self.obj_mesh.name, self.grid_size)

    def __repr__(self):
        return self.__str__()

    def _project_on_box(self, pt):
        """Project the point on the box if it is outside the box
        @param pt - the x,y,z point in the sdf coordinate system
        @returns distance to box, location on box, in the sdf coordinate system"""
        pt_on_box = self.bbox.project_on_box(pt)
        raise NotImplementedError

    def _eval_signed_distance_func(self, pt):
        """ Evaluate the signed distance function - assumes the point is inside the bbox
        @param pt - the x,y,z point in the sdf coordinate system
        @returns signed distance in the sdf coordinate system """
        raise NotImplementedError

    def _eval_signed_distance_func_gradient(self, pt):
        """ Evaluate the signed distance function gradient - assumes the point is inside the bbox
        Might want to do a combined calculation (point/gradient)
        @param pt - the x,y,z point in the sdf coordinate system
        @returns gradient in the sdf coordinate system """
        raise NotImplementedError

    def is_inside_box(self, pt):
        """Is the point inside the bounding box?
        @param pt - the x,y,z point in the sdf coordinate system
        @returns true/false
        """
        return self.bbox.is_inside_box(pt)

    def _calc_distance_to_surface(self, pt):
        """ Calculates the distance to the surface in the sdf coordinate system
        @param pt - the x,y,z point in the sdf coordinate system
        @returns signed distance in the sdf coordinate system """
        if self.is_inside_box(pt):
            return self._eval_signed_distance_func(pt)

        dist_to_box, pt_on_box = self._project_on_box(pt)
        return dist_to_box + self._eval_signed_distance_func(pt_on_box)

    def _calc_gradient(self, pt):
        """ Calculates the gradient to the surface in the sdf coordinate system
        @param pt - the x,y,z point in the sdf coordinate system
        @returns gradient in the sdf coordinate system """
        if self.is_inside_box(pt):
            return self._eval_signed_distance_func_gradient(pt)

        _, pt_on_box = self._project_on_box(pt)
        return self._eval_signed_distance_func_gradient(pt)

    def calc_distance_to_surface(self, pt_in_mesh):
        """ Signed distance function
         @param pt_in_mesh - point in the MESH'S coordinate system
         @returns distance in mesh coordinate system"""
        pt_in_sdf = self.from_mesh_to_sdf.transform(pt_in_mesh)
        res = self._calc_distance_to_surface(pt_in_sdf)

        # undo scaling
        return res / self.from_mesh_to_sdf.scaling[0]

    def calc_gradient(self, pt_in_mesh):
        """ Signed distance function
         @param pt_in_mesh - point in the MESH'S coordinate system
         @returns gradient in mesh coordinate system"""
        # Note: Might want to cache this matrix
        pt_in_sdf = self.from_mesh_to_sdf.transform(pt_in_mesh)
        res = self._calc_gradient(pt_in_sdf)

        # undo scaling - TODO, fix so it works with whatever type we use
        return res / self.from_mesh_to_sdf.scaling[0]

    def set(self, obj: ObjectBase, padding=0.1, grid_size=10, b_cache=True):
        """ Set up the signed distance function and the transform
        @param obj: Object base class
        @param padding: Percentage of overall cube to use as padding
        @param grid_size: Number of grid cells to use in max size direction
        @param b_cache: If the cached file exists, use it"""
        # Set up the bounding box
        # Set up the Coordinate System Transform
        # Fill up the bounding box
        # (eventually) see if we've cached this one
        raise NotImplementedError

    def file_cache(self):
        """Write to the cached file"""
        raise NotImplementedError


if __name__ == "__main__":

    my_sdf = SignedDistanceFc()
    my_obj = ObjectBase("Test")

    my_sdf.set(my_obj)
