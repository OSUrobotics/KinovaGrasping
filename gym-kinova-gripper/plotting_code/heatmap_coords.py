import numpy as np
from pathlib import Path

# Add heatmap initial object coordinates to dictionary, filter by success/fail
# and orientation type, then save in appropriate directory

def sort_coords_by_region(dict_list):
    """ Sort the input coordinates dictionaries by region and return the updated dictionaries with a new 'region' entry
    based on the global object coordinate location """
    # Definition of coordinate sampling regions within the graspable space within the hand
    grasping_regions = {"extreme_left": [-0.12, -0.055], "mid_left": [-0.055, -0.03], "center": [-0.03,0.03], "mid_right": [0.03,0.055], "extreme_right":[0.055,0.12]}

    for d in dict_list:
        coords = d["local_obj_coords"]
        if grasping_regions["extreme_left"][0] <= coords[0] <= grasping_regions["extreme_left"][1]:
            d["local_coord_region"] = "extreme_left"
        elif grasping_regions["mid_left"][0] < coords[0] <= grasping_regions["mid_left"][1]:
            d["local_coord_region"] = "mid_left"
        elif grasping_regions["center"][0] < coords[0] < grasping_regions["center"][1]:
            d["local_coord_region"] = "center"
        elif grasping_regions["mid_right"][0] <= coords[0] < grasping_regions["mid_right"][1]:
            d["local_coord_region"] = "mid_right"
        elif grasping_regions["extreme_right"][0] <= coords[0] <= grasping_regions["extreme_right"][1]:
            d["local_coord_region"] = "extreme_right"

    return dict_list

def sort_coords_by_success(all_hand_object_coords, shape, orientation, frame="local"):
    """Sort the hand-object pose coordinates by success or failure per shape and per orientation"""
    # Just take out the successful local coordinates for plotting
    success_coords = [d[frame + "_obj_coords"] for d in all_hand_object_coords if d["success"] is True and d["shape"] == shape and d["orientation"] == orientation]
    fail_coords = [d[frame + "_obj_coords"] for d in all_hand_object_coords if d["success"] is False and d["shape"] == shape and d["orientation"] == orientation]

    return success_coords, fail_coords

# Coordinate Saving Step 1: Filter heatmap coordinates by success/fail, orientation, and shape
def sort_and_save_heatmap_coords(all_hand_object_coords, requested_shapes, requested_orientations, episode_num, saving_dir, frame="local"):
    """ Save heatmap coordinates split by success and orientation """
    for orientation in requested_orientations:
        for shape in requested_shapes:
            coord_save_dir = saving_dir + "/" + orientation + "/" + shape + "/" + frame
            success_coords, fail_coords = sort_coords_by_success(all_hand_object_coords, shape, orientation, frame)
            success_x, success_y, success_z, fail_x, fail_y, fail_z, total_x, total_y, total_z = coords_dict_to_array(success_coords,fail_coords)
            save_coordinates(success_x, success_y, success_z, coord_save_dir, "/success", episode_num)
            save_coordinates(fail_x, fail_y, fail_z, coord_save_dir, "/fail", episode_num)
            save_coordinates(total_x, total_y, total_z, coord_save_dir, "/total", episode_num)

            print("Writing to heatmap coordinate info file...")
            f = open(coord_save_dir + "/heatmap_info.txt", "w")
            save_text = "Heatmap Coords \nSaved at: "+coord_save_dir
            shape_text = "\nShape: " + str(shape)
            orientation_text = "\n"+str(orientation).capitalize()+" Orientation\n# Success: "+str(len(success_x))+"\n# Fail: "+str(len(fail_x))+"\n"
            text = save_text + shape_text + orientation_text
            f.write(text)
            f.close()


# Coordinate Saving Step 2: Convert coordinate dictionaries to numpy arrays for saving
def coords_dict_to_array(success_coords, fail_coords):
    """ Convert coordinate dictionary to numpy array for saving """
    success_x = np.array([])
    success_y = np.array([])
    success_z = np.array([])
    fail_x = np.array([])
    fail_y = np.array([])
    fail_z = np.array([])
    total_x = np.array([])
    total_y = np.array([])
    total_z = np.array([])

    if len(success_coords) > 0:
        success_x = np.array([coords[0] for coords in success_coords])
        success_y = np.array([coords[1] for coords in success_coords])
        success_z = np.array([coords[2] for coords in success_coords])
        total_x = np.append(total_x,success_x)
        total_y = np.append(total_y,success_y)
        total_z = np.append(total_z,success_z)
    if len(fail_coords) > 0:
        fail_x = np.array([coords[0] for coords in fail_coords])
        fail_y = np.array([coords[1] for coords in fail_coords])
        fail_z = np.array([coords[2] for coords in fail_coords])
        total_x = np.append(total_x,fail_x)
        total_y = np.append(total_y,fail_y)
        total_z = np.append(total_z,fail_z)

    return success_x, success_y, success_z, fail_x, fail_y, fail_z, total_x, total_y, total_z

# Coordinate Saving Step 3: Save heatmap coordinates by orientation type
def save_coordinates(x,y,z,file_path,filename,episode_num):
    """ Save heatmap initial object position x,y coordinates
    x: initial object x-coordinate
    y: initial object y-coordinate
    z: initial object z-coordinate
    filename: Location to save heatmap coordinates to
    """
    # Ensure file path is created if it doesn't exist
    coord_save_path = Path(file_path)
    coord_save_path.mkdir(parents=True, exist_ok=True)

    ep_str = ""
    if episode_num is not None:
        ep_str = "_"+str(episode_num)

    np.save(file_path + filename+"_x"+ep_str, x)
    np.save(file_path + filename+"_y"+ep_str, y)
    np.save(file_path + filename + "_z" + ep_str, z)