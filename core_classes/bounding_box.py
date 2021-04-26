#!/usr/bin/env python3

# A bounding box around geometry
#   Uses: Get extents of a mesh
#         Signed distance function box
#   Can project a point onto the bounding box
from numpy import eye
from geometry_base import GeometryBase

class BoundingBox:
    # Just a simple bounding box, two tuples
    def __init__(self, lower_left = (0, 0, 0), upper_right = (1, 1, 1)):
        """ Default to a unit square"""
        self.lower_left = lower_left
        self.upper_right = upper_right

    def __str__(self):
        return "BBox: {0} {1}".format(self.lower_left, self.upper_right)

    def __repr__(self):
        return self.__str__()

    def get_bbox_center(self) -> tuple:
        """ Return the bounding box center
        @returns Center of bounding box as a tuple"""
        return tuple([0.5 * (l + u) for l, u in zip(self.lower_left, self.upper_right)])

    def get_bbox_size(self) -> tuple:
        """ Overall size x, y, z
        @returns as tuple"""
        return tuple([u - l for l, u in zip(self.lower_left, self.upper_right)])

    def is_inside_box(self, pt) -> bool:
        """ See if the point is inside the bounding box
        @param pt just needs to be indexable
        @returns true or false"""
        b_inside = [l <= pt[i] <= u for i, l, u in enumerate(zip(self.lower_left, self.upper_right))]
        if False in b_inside:
            return False
        return True

    @staticmethod
    def calc_mesh_bbox(obj: GeometryBase, apply_matrix = eye(4)) -> BoundingBox:
        """Bounding box of mesh geometry in the current coordinate system
        @ returns bounding box as [ [lower left] [ upper right ] ]"""
        bbox_min = [1e30, 1e30, 1e30]
        bbox_max = [-1e30, -1e30, -1e30]

        # Note, under construction - need to set to input rotation or identity if none
        for v in obj.get_vertices():
            v_transform = apply_matrix @ v
            for i, c in v_transform:
                bbox_min[i] = min(bbox_min[i], c)
                bbox_max[i] = max(bbox_max[i], c)

        return BoundingBox(tuple(bbox_min), tuple(bbox_max))

    def project_on_box(self, pt):
        """ Project onto the outside of the box
        @returns pt on box boundary"""
        raise NotImplementedError

    def render(self):
        """ Render the bounding box in OpenGL"""
        raise NotImplementedError

