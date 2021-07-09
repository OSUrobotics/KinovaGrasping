#!/usr/bin/env python3

# Naming of coordinate system transforms, naming of coordinate systems, and standardized bounding boxes/drawing
# CoordinateSystemBase: Use to name a location/orientation on a given piece of mesh/geometry.
#  EG, the palm of a hand, the base of an "upright" orientation of an object, the center of a table,
#    the location of an aruco marker, etc.
# CoordinateSystemTransformBase: Use to make matrices that take objects from one coordinate system to another
#  The from and to coordinate systems are explicitly named to support debugging/sanity checking
#  Also has support tools for building matrices in common ways (eg, a translation and a rotation
#  Bounding box is the mesh/geometry BEFORE the transformation - can always apply matrix transform to get resulting bbox
#
# Assumptions for all coordinate systems is that z is up and x is left-right, y is in-out

from scipy.spatial.transform import Rotation as R
from numpy import array as nparray
from numpy import pi, eye
from numpy.linalg import norm
from geometry_base import GeometryBase
from bounding_box import BoundingBox


class CoordinateSystem:
    # Assumption is that the coordinate system is based off of the geometry's default location as read in from the mesh
    #   Location: Where on the mesh
    #   Orientation: Rotate the coordinate system
    #   Scale: Default is to scale so the vectors are the same size as mesh

    # Some common types of coordinate systems
    coordinate_system_type = {"world":{"origin"},
                              "hand":{"origin", "xmlOrURDF", "centered", "wrist", "handspan", "distal"},
                              "object":{"origin", "xmlOrURDF", "centered", "base", "center", "grasp"},
                              "environment":{"origin", "xmlOrURDF", "center", "aruco"},
                              "arm":{"origin", "xmlOrURDF", "base", "endeffector"},
                              "signedDistanceFunction":{"origin", "center"},
                              "camera":{"origin", "xmlOrURDF", "at", "aruco"}}


    def __init__(self, in_type=("none", "none"), in_name="none",
                 in_translation=(0, 0, 0), in_rotation=eye(), in_scale=(1, 1, 1)):
        """ In the object mesh's coordinate system, scale, then rotate, then translate the x,y,z coordinate system
           rotation: How to rotate (1,0,0) (0,1,0) (0,0,1) to be the x, y, z for this coordinate system
        @param my_type: tuple of strings from coordinate_system_type
        @param my_name: An additional name
        @param in_translation: Where to put the center of the coordinate system
        @param in_rotation: Scipy Rotation object. If not given, will use identity transform. How to rotate the coordinate system before moving it
        @param in_scale: How to scale the coordinate system before rotating and translating it"""
        self.check_type(in_type)
        self.coord_type = in_type
        self.name = in_name
        self.translation = in_translation
        self.rotation = in_rotation
        self.scale = in_scale

    def __str__(self):
        """TODO print out coordinate system information"""
        raise NotImplementedError
        #return "Coord Sys: {} to {}".format(self.from_coord_sys. self.to_coord_sys)

    def __repr__(self):
        """print out coordinate system information"""
        return self.__str__()

    @staticmethod
    def check_type(in_type):
        if in_type[0] not in CoordinateSystem.coordinate_system_type:
            raise KeyError("Could not find {0} coordinate system type".format(in_type[0]))
        else:
            if in_type[1] not in CoordinateSystem.coordinate_system_type[in_type[0]]:
                raise KeyError("Could not find {0} coordinate system sub type".format(in_type[1]))

        return True

    def get_matrix(self):
        """ Turn the coordinate system into a matrix that takes the origin x,y,z to the given point"""
        mat = eye(4)
        mat[0:3, 0:3] = self.rotation.to_matrix()
        for i, s in self.scale:
            mat[0:3, i] = mat[0:3, i] * s
        mat[3, 0:3] = nparray(self.translation).transpose()
        return mat

    def get_inverse_matrix(self):
        """ TODO Reverse the matrix construction"""
        mat = eye(4)
        return mat

    def render(self):
        """ TODO Render self as a point and 3 vectors"""
        pass


class CoordinateSystemTransform:
    # The following two are for setting a coordinate transform based on the geometry
    # Where the base of the coordinate system is
    base_location_type = {"current", "center", "bottom_middle", "lower_left"}

    # Overall scale normalization
    scale_normalization_type = {"no_scale", "unit_square", "unit_square_plus_pad", "percentage" }

    def __init__(self, from_coord_sys=("world", "origin"), to_coord_sys=("world", "origin"), xform = eye(4)):
        """A matrix transform from one coordinate system to another
           Explicitly name the from and to coordinate systems
           @param from_coord_sys: one of coord_system_type
           @param to_coord_sys: one of coord_system_type
           @param xform: The actual matrix - to create a specific transform, use one of the create methods"""
        CoordinateSystem.check_type(from_coord_sys)
        CoordinateSystem.check_type(to_coord_sys)

        # The from and the to shouldn't change
        # use the set methods to set the xform for specific cases
        self.from_coord_sys = from_coord_sys
        self.to_coord_sys = to_coord_sys
        self.xform = xform

        # Keep the from bounding box as a stand-in for the geometry
        self.bbox_from = BoundingBox()

    def __str__(self):
        """TODO print out coordinate system information"""
        raise NotImplementedError
        #return "Coord Sys: {} to {}".format(self.from_coord_sys. self.to_coord_sys)

    def __repr__(self):
        """print out coordinate system information"""
        return self.__str__()

    def calc_distance(self, size_normalize=1):
        """Calculate how much the transform changes the object TODO
        @param size_normalize - what is considered a 'unit' translation
        @returns orienation distance (as measured by quaternion dot product), and Euclidean distance, and combined"""
        rot_err = norm(self.orientation.as_quat()) / (pi/4)
        trans_err = norm(nparray(self.translation)) / size_normalize
        return 0.5 * (rot_err + trans_err), rot_err, trans_err

    @staticmethod
    def calc_centered_from_geometry(obj_geom: GeometryBase, orientation:R = R.identity()):
        """Define a transform, based on the object's geometry, that puts the center at the origin
            and scales it to fit in a unit cube
           Note: Scaling happens *after* the rotation
        @param obj_geom - the actual geometry
        @param orientation - re-orient the object
        @return A transform that centers the object"""

        # TODO: should read the from from the geometry class
        #   will be object/hand, "origin", object/hand, "centered"
        center_mesh = CoordinateSystemTransform("mesh", "centered")
        center_mesh.bbox_from = BoundingBox.calc_mesh_bbox(obj_geom, orientation)

        mesh_center = center_mesh.bbox_from.get_bbox_center()
        center_mesh.translation = (-mesh_center[0], -mesh_center[1], -mesh_center[2])
        center_mesh.orientation = orientation
        scl = 1 / max(center_mesh.bbox_from.get_bbox_size())
        center_mesh.scaling = (scl, scl, scl)

        return center_mesh

    @staticmethod
    def calc_transform_from_centered(self, centering_transform, origin_location = "center", scale_normalization = "no_scale", scl_perc = (1, 1, 1)):
        """Define a transform, based on the object's centered geometry, that puts the center and scale as given
           Note: Re-scaling happens *before* translation
        @param centering_transform - the CoordinateSystemTransformBase that centered the object
        @param origin_location - Which point should be at 0,0,0 after the transform?
        @param scale_normalization - set the scale to be unit square OR a percentage of scale
        @param scl_perc - optional parameter to set the overall scale size in x, y, z as a percentage
            if one number given, applys that to all dimensions
        @return The transform that takes the object from the centered position to the new one"""
        if origin_location not in CoordinateSystemTransform.base_location_type:
            raise ValueError("Origin location should be one of base_location_type, got {0}".format(origin_location))
        if scale_normalization not in CoordinateSystemTransform.scale_normalization_type:
            raise ValueError("Scale normalization should be one of of scale normalization type, got {0}".format(scale_normalization))

        # TODO Check that the bounding box in the centering transform is actually centered
        # TODO Create the from and to based on type type from centering transform (hand/object)
        # TODO Create the matrix that scales the object and translates it, based on the inputs
        center_mesh = CoordinateSystemTransform("TODO")
        return center_mesh

    def transform(self, pt):
        """Transform the point from the From coord sys to the To one
        @param pt: 3 numbers, nparray, list, tuple
        @returns 3 numbers in ??"""
        # TODO Manually apply the matrix
        raise NotImplementedError

    def transform_back(self, pt):
        """Transform the point from the To coord sys to the From one
        @param pt: 3 numbers, nparray, list, tuple
        @returns 3 numbers in ??"""
        # TODO Manually apply the inverse matrix
        raise NotImplementedError

    def get_to_bbox(self) -> BoundingBox:
        """ Multiply the from bbox by the matrix
          Note that this will be a poor approximation if there's a non 90-rotation involved
        @returns Output bounding box"""

        return self.bbox_from.calc_transformed_bbox(self.get_matrix())

    def get_matrix(self) -> nparray:
        """ Return the transform
        @returns: np array representing a matrix"""
        return self.xform()

    def get_inverse_matrix(self) -> nparray:
        """ -translation * rotation transposed * 1/scaling
        @returns: np array representing a matrix"""
        trans = nparray((4, 4))
        # Set translation and rotation
        raise NotImplementedError
        #return trans

    def __matmul__(self, other: CoordinateSystemTransform):
        """ Matrix multiply AND check that previous and next match up
        @other Another coordinate system transform
        @return A coordinate system transform"""
        if self.to_coord_sys is not other.from_coord_sys:
            raise ValueError("To and from don't match {} {}".format(self, other))
        combined_transform = CoordinateSystemTransform(self.from_coord_sys, other.to_coord_sys)
        combined_transform.xform = self.xform @ other.xform
        return combined_transform

    @staticmethod
    def combine_transforms(seq_transforms) ->nparray:
        """ Create a new trasnform from the sequence of transforms
        Double checks that the sequence is valid  (froms match tos)
        @param seq_transforms - iterable of CoordinateSystemTransformBase transforms
        @returns 4x4 matrix"""
        m = eye(4)

        # TODO make a coordinateSystemTransform
        source_coord_sys = seq_transforms[0].from_coord_sys
        prev = None
        for c in seq_transforms:
            m = c.get_matrix @ m

            if prev:
                if prev.to_coord_sys is not c.from_coord_sys:
                    raise ValueError("Coordinate system transform from-to don't match {0} {1}".format(prev, c) )
            prev = c

        return m, (source_coord_sys, prev.to_coord_sys)

    def set_from_coordinate_sysetm(self, coord_sys: CoordinateSystem):
        """ TODO Make (and store) the actual matrix
             Make sure from/to are set up correctly"""
        raise NotImplementedError

    def set_from_xml(self, im_not_sure):
        """ TODO Not quite sure what this looks like - but get the transform from the xml file
        NOTE: Might put on object geometry instead"""
        raise NotImplementedError

    def construct_json(self) -> dict:
        """TODO Construct a dictionary/array structure that can be written to a json file
          - By default this just takes and makes pairs of all the self variable names and their values"""
        raise NotImplementedError

    @staticmethod
    def read_from_json(json_data):
        """TODO Take in json data and construct a CoordinateSystemBase
        @param json_data: Dictionary/array returned from json read
        @param relative_to - pass in if known, or None if assumed global
        @returns - CoordinateSystemBase """
        raise NotImplementedError

    def render(self, b_draw_from_bbox = True, b_draw_to_bbox = True, b_draw_to_coord_sys = True):
        """ TODO Draw the from and to bounding boxes as well as where the origin/vecs to go
        Assumption is that you've rendered the object and you want to check that the transform is in the
        correct location"""
        raise NotImplementedError

    def check_valid_pre(self, pre_cst):
        """ Check that the to matches our from
        @param pre_cst - the CoordinateSystemTransformBase BEFORE this one"""
        if pre_cst.to_coord_sys is not self.from_coord_sys:
            raise ValueError("Coordinate systems do NOT match {0} {1}".format(pre_cst.to_coord_sys, self.from_coord_sys))
        return True


if __name__ == "__main__":

    my_sys = CoordinateSystemTransform(("world", "origin"))
