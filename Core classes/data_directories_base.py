#!/usr/bin/env python3

# We have a LOT of data. This class:
#   Handles matching data folders to class hierarchy names
#   Handles matching data folders to instance names
#   Handles filename/extension types
#   Handles naming conventions

from pathlib import Path

class DataDirectoryBase:
    # By default, all data will go in the source code folder in a folder called "data"
    data_location = Path("./Data/")

    # These objects can be built from code
    defined_object_names = {"Cone", "Cube", "Hour_glass", "Cylinder", "Bowl"}
    # These are the ycb objects
    ycb_object_names = {} # Read this one from a file

    # Defined hand names that we've used/built - might eventually fill in from the data directories themselves
    defined_hand_names = {"2V2", "2V3"}
    # Will need to fix this with specific types
    existing_hand_names = {"Barrett", "Kinova", "Robotiq"}

    # Environmental objects
    enviroments = {"Table", "Reset_mechanism", "Door", "Drawer", "Apple"}

    def __init__(self, cls=None, instance_name=None):
        self.path_name = Path(self.construct_path_name(cls, instance_name))

    @staticmethod
    def construct_path_name(cls, instance_name=None):
        """ Create a path name from the class variable names and an (optional) instance name
        @param cls - the Class name of the instance being created
        @param instance_name - one of the names from the sets given above"""
        if instance_name is not None:
            path_name = "/" + instance_name
        else:
            path_name = "/"

        traverse_cls = cls
        while super(traverse_cls):
            path_name = str(traverse_cls) + "/" + path_name
            traverse_cls = cls.super()

        path_name = DataDirectoryBase.data_location + path_name
        return path_name

    def check_path_exists(self, b_make_path=False):
        """ Check to see if the path exists, and possibly make it
          Throws a warning if the path up to the last folder does not exist
          @param b_make_path Make the path y/n"""
        if self.path_name.exists():
            return True

        if not b_make_path:
            raise FileNotFoundError("Path name {0} does not exist".format(self.path_name))

        if self.path_name.parent().exists():
            mkdir(self.path_name)




if __name__ == "__main__":


