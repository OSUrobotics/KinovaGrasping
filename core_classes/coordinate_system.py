#!/usr/bin/env python3

# Naming of coordinate system transforms and standardized bounding boxes/drawing
#  Transform is specified by a from and a to (named) coordinat system
#  Actual transform consist of a translation, rotation, scaling (M = S R T object)
#  Bounding box is the mesh/geometry AFTER the transformation
#  Assumptions for all coordinate systems is that z is up and x is left-right, y is in-out

from scipy.spatial.transform import Rotation as R
from numpy import array as nparray
from numpy import pi
from numpy.linalg import norm
from geometry_base import GeometryBase
from bounding_box import BoundingBox


class CoordinateSystemTransformBase:
    # The following two are for setting a coordinate transform based on the geometry
    # Where the base of the coordinate system is
    base_location_type = {"current", "center", "bottom_middle", "lower_left"}

    # Overall scale normalization
    scale_normalization_type = {"no_scale", "unit_square", "unit_square_plus_pad", "percentage" }

    # What kind of coordinate system transform is it? See Miro file and doc for examples/explanations
    coord_system_type = {"world", "mesh", "centered",
                         "xml_urdf",
                         "signed_distance_function",
                         "object_base", "object_center", "hand_attachment", "hand_palm",
                         "arm_base", "arm_end_effector",
                         "camera"}

    def __init__(self, from_coord_sys="mesh", to_coord_sys="world",
                 translation=(0, 0, 0), scaling=(1, 1, 1), orientation: R=R.identity()):
        """Essentially a translation followed by a rotation followed by a scaling
           Explicitly name the from and to coordinate systems
           Orientation: How to rotate (1,0,0) (0,1,0) (0,0,1) to be the x, y, z for this coordinate system
           @param from_coord_sys: one of coord_system_type
           @param to_coord_sys: one of coord_system_type
           @param translation: How much to translate the object by
           @param scaling: tupleof scale values. Default is scale by 1
           @param orientation: Scipy Rotation object. If not given, will use identity transform"""
        if from_coord_sys not in CoordinateSystemTransformBase.coord_system_type:
            raise KeyError("Could not find from coord systemm name {}".format(from_coord_sys))
        if to_coord_sys not in CoordinateSystemTransformBase.coord_system_type:
            raise KeyError("Could not find to coord systemm name {}".format(to_coord_sys))

        # In the end, just a translation, rotation, scaling
        # ... but record the from and to type
        self.from_coord_sys = from_coord_sys
        self.to_coord_sys = to_coord_sys
        self.translation = translation
        self.scaling = scaling
        self.orientation = orientation

    def __str__(self):
        """print out coordinate system information"""
        raise NotImplementedError
        #return "Coord Sys: {} to {}".format(self.from_coord_sys. self.to_coord_sys)

    def __repr__(self):
        """print out coordiante system information"""
        return self.__str__()

    def calc_distance(self, size_normalize=1):
        """Calculate how much the transform changes the object TODO
        @param size_normalize - what is considered a 'unit' translation
        @returns orienation distance (as measured by quaternion dot product), and Euclidean distance, and combined"""
        rot_err = norm(self.orientation.as_quat()) / (pi/4)
        trans_err = norm(nparray(self.translation)) / size_normalize
        return 0.5 * (rot_err + trans_err), rot_err, trans_err

    def set_from_xml(self, im_not_sure):
        """ Not quite sure what this looks like - but get the transform from the xml file"""
        raise NotImplementedError

    @staticmethod
    def calc_centered_from_geometry(obj_geom: GeometryBase, orientation:R = R.identity()):
        """Define a transform, based on the object's geometry, that puts the center at the origin
            and scales it to fit in a unit cube
           Note: Scaling happens *after* the rotation
        @param obj_geom - the actual geometry
        @param orientation - re-orient the object
        @return Transform to center"""

        bbox_mesh = BoundingBox.calc_mesh_bbox(obj_geom, orientation)

        center_mesh = CoordinateSystemTransformBase("mesh", "centered")

        mesh_center = bbox_mesh.get_bbox_center()
        center_mesh.translation = (-mesh_center[0], -mesh_center[1], -mesh_center[2])
        center_mesh.orientation = orientation
        scl = 1 / max(bbox_mesh.get_bbox_size())
        center_mesh.scaling = (scl, scl, scl)

        return center_mesh

    def calc_transform_from_centered(self, obj_geom: GeometryBase,
                                     origin_location = "center", orientation:R = R.identity(), scale_normalization = "no_scale", scl_perc = (1, 1, 1)):
        """Define a transform, based on the object's geometry, that puts the center and scale as given
           Note: Re-scaling happens *after* the rotation
        @param obj_geom - the actual geometry
        @param origin_location - Which point should be at 0,0,0 after the transform?
        @param orientation - re-orient the object
        @param scale_normalization - set the scale to be unit square OR a percentage of scale
        @param scl_perc - optional parameter to set the overall scale size in x, y, z as a percentage
            if one number given, applys that to all dimensions
        @return One transform if origin location is center, otherwise, two"""
        if origin_location not in CoordinateSystemTransformBase.base_location_type:
            raise ValueError("Origin location should be one of base_location_type, got {0}".format(origin_location))
        if scale_normalization not in CoordinateSystemTransformBase.scale_normalization_type:
            raise ValueError("Scale normalization should be one of of scale normalization type, got {0}".format(scale_normalization))

        bbox_mesh = BoundingBox.calc_mesh_bbox(obj_geom)

        center_mesh = CoordinateSystemTransformBase("Mesh", "")

        # May actually be *two* transforms - one to take the object to the center, one
        # This will be one bbox center, etc TODO
        self.translation = self.bbox.get_bbox_center

        # Use this orientation
        self.orientation = orientation

        # Calculate scale needed based off of bounding box sizes TODO
        self.scaling = (1, 1, 1)

        # Last but not least, apply the translation/scale to the bbox
        return

    def set_from_concatenation(self, seq_of_transforms):
        """Given a sequence of transforms, multiply them together to make one
        @param seq_of_transforms: Any iterable over CoordinateSystemTransformBase objects"""
        raise NotImplementedError

    def transform(self, pt):
        """Transform the point from the From coord sys to the To one
        @param pt: 3 numbers, nparray, list, tuple
        @returns 3 numbers in ??"""
        # Manually transform and scale
        raise NotImplementedError

    def transform_back(self, pt):
        """Transform the point from the To coord sys to the From one
        @param pt: 3 numbers, nparray, list, tuple
        @returns 3 numbers in ??"""
        # Manually transform and scale
        raise NotImplementedError

    def get_matrix(self) -> nparray:
        """ Scale * rotation * translation
        @returns: np array representing a matrix"""
        trans = nparray((4, 4))
        # Set translation and rotation
        raise NotImplementedError
        #return trans

    def get_inverse_matrix(self) -> nparray:
        """ -translation * rotation transposed * 1/scaling
        @returns: np array representing a matrix"""
        trans = nparray((4, 4))
        # Set translation and rotation
        raise NotImplementedError
        #return trans

    def construct_json(self) -> dict:
        """Construct a dictionary/array structure that can be written to a json file
          - By default this just takes and makes pairs of all the self variable names and their values"""
        raise NotImplementedError

    @staticmethod
    def read_from_json(json_data):
        """Take in json data and construct a CoordinateSystemBase
        @param json_data: Dictionary/array returned from json read
        @param relative_to - pass in if known, or None if assumed global
        @returns - CoordinateSystemBase """
        raise NotImplementedError

    def render_to_coordinate_system(self):
        """ Draw the coordinate system as a sphere for the transform point and scaled vectors for the x, y, z
        Assumption is that you've rendered the object and you want to check that the transform is in the
        correct location"""
        raise NotImplementedError

    def render_bbox(self, render_from_bbox = False):
        """ Render the bounding box either before or after the transform
        @param render_from_bbox assumption is that you want to render the to bbox"""
        bbox = self.bbox
        if render_from_bbox:
            # transform bbox
            raise NotImplementedError
        bbox.render()

    def check_valid_pre(self, pre_cst):
        """ Check that the to matches our from
        @param pre_cst - the CoordinateSystemTransformBase BEFORE this one"""
        if pre_cst.to_coord_sys is not self.from_coord_sys:
            raise ValueError("Coordinate systems do NOT match {0} {1}".format(pre_cst.to_coord_sys, self.from_coord_sys))
        return True


if __name__ == "__main__":

    my_sys = CoordinateSystemTransformBase("world")
