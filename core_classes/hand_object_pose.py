#!/usr/bin/env python3

# Given a hand and an object, define a transform between them
#  Handles checking for hand-object collision
# Requires: A hand and an object
#  Transforms are the hand's palm/handspan relative to the object's pose
#   Can string together with object pose to get world coordinate system pose for hand
#   Can string together with wrist-palm pose to get desired end effector pose (for IK)
# Derived class adds noise to a given HandObjectPose

from coordinate_system import CoordinateSystemBase, CoordinateSystemTransformBase
from object_geometry_base import ObjectGeometryBase
from hand_geometry_base import HandGeometry
from data_directories_base import DataDirectoryBase


class HandObjectPoseBase(DataDirectoryBase):
    """ """
    pose_location_type = {"side":{"x pos", "y pos", "x neg", "y neg", "xy ang"},
                          "top":{"x axis", "y axis", "xy diag", "xy ang"},
                          "fortyfive":{"x pos y pos", "x pos y neg", "x neg y neg", "x neg y pos", "xy ang"},
                          "sampled":{"sdf", "spherical"},
                          "invalid":{}}

    def __init__(self,
                 hand_obj: HandGeometry, obj_obj: ObjectGeometryBase, env_obj=None):
        """ What type of hand-object pose is this?
        @param loc_type one of pose_location_type
        @param preshape_type one of pose_preshape_type
        @param noise_type one of pose_nosie_type"""

        self.location_type = "invalid"
        self.xy_ang = 0
        self.preshape_type = "cylindrical"
        self.noise_type = "none"

        # Geometry this is defined on
        self.hand = hand_obj
        self.obj = obj_obj
        self.env = env_obj

        # We need this transform to define how the hand is oriented wrt to the object in the "base" coordinate system
        self.pose = CoordinateSystemBase(("object", "grasp"))

    def __str__(self):
        """ TODO: Build string from hand and object type and location type"""
        return "Data path: {0}".format(self.location_type)

    def __repr__(self):
        return self.__str__()

    def set_invalid(self):
        self.location_type = ("invalid", "")

    def set_defined_pose(self, loc_type, xy_ang=0, dist=0.5):
        """ Set one of the pre-define hand pose locations
        Distance is the percentage of the hand span depth to place the object surface at
        @param: loc_type - one of pose_location_type
        @param: xy_ang - optional angle parameter
        @param: dist - optional distance of object surface to palm center parameter. If this distance fails, will try others
        @param: returns True if successful (no intersection) """
        if loc_type not in HandObjectPoseBase.pose_location_type:
            raise KeyError("Pose location {0} is not one of pose_location_type".format(loc_type))

        self.location_type = loc_type
        self.xy_ang = xy_ang

        # TODO Essentially a bunch of if statements, positioning the hand relative to the object in the given direction
        # Try a variety of distances

        if self.is_intersecting():
            self.set_invalid()
            return False
        raise NotImplementedError

    def set_spherical_pose(self, pt_on_sphere=(1, 0, 0), center_pt_bbox=(0.5, 0.5, 0.5)):
        """TODO position the hand on a vector from the object's bbox center pointed at the center
        @param pt_on_sphere: The x,y,z point on the sphere
        @param center_pt_bbox: Where to put the center point of the bounding box (percentage of bbox, default mid)
        @param: returns True if successful (no intersection) """
        self.location_type = ("sampled", "spherical")
        if self.is_intersecting():
            self.set_invalid()
            return False
        raise NotImplementedError

    def set_sdf_pose(self, pt_in_xyz=(1, 0, 0), dir_in_vxvyvz=(-1, 0, 0), , dir_up_vxvyvz=(0, 1, 0)):
        """TODO position the hand at the given point in the signed distance function bounding box, pointed in the
           given direction
        @param pt_in_xyz: The x,y,z point in the object's bbox
        @param dir_in_vxvyvz: Direction to point the palm at
        @param dir_up_vxvyvz: Direction to point the palm Width direction at
        @param: returns True if successful (no intersection) """
        self.location_type = ("sampled", "sdf")
        if self.is_intersecting():
            self.location_type = ("invalid", "")
            return False
        raise NotImplementedError

    def is_intersecting(self):
        """ TODO Check if hand in this pose intersects the object or the environment
        @returns True/False"""
        raise NotImplementedError

    def render(self, coord_sys=("world", "origin")):
        """TODO Draw object bounding box, coordinate system for hand pose
         Depending on coordinate system, apply tye appropriate matrix first
         @param coord_sys: One of world, hand, or object"""
        raise NotImplementedError


class HandObjectPoseNoise(HandObjectPoseBase):
    """ Add noise to an existing hand-object pose"""
    pose_noise_type = {"position only", "orientation only", "position and orientation", "none"}

    def __init__(self, parent_pose: HandObjectPoseBase):
        """ What pose are we adding noise to
        @param parent_pose """

        super(HandObjectPoseBase, self).__init__(parent_pose.hand, parent_pose.obj, parent_pose.env)
        self.location_type = "invalid"
        self.xy_ang = 0
        self.preshape_type = "cylindrical"
        self.noise_type = "none"

        # Geometry this is defined on
        self.hand = hand_obj
        self.obj = obj_obj

        # We need this transform to define how the hand is oriented wrt to the object in the "base" coordinate system
        self.pose = CoordinateSystemBase(("object", "grasp"))

    pose_type = {{"side":{"x pos", "y pos", "}, "top", "fortyfive"}
