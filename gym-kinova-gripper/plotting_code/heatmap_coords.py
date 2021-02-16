import numpy as np
from pathlib import Path

# Add heatmap initial object coordinates to dictionary, filter by success/fail
# and orientation type, then save in appropriate directory


def add_heatmap_coords(success_coords, fail_coords, orientation, obj_coords, success):
    """ Add object coordinates to success/failed coordinates dictionary for heatmaps  """
    # Get object coordinates, transform to array
    x_val = obj_coords[0]
    y_val = obj_coords[1]
    x_val = np.asarray(x_val).reshape(1)
    y_val = np.asarray(y_val).reshape(1)

    # Heatmap postion data - get starting object position and mark success/fail based on lift reward
    if success:
        # Append initial object coordinates to Successful coordinates array
        success_coords["x"] = np.append(success_coords["x"], x_val)
        success_coords["y"] = np.append(success_coords["y"], y_val)
        success_coords["orientation"] = np.append(success_coords["orientation"], orientation)
    else:
        # Append initial object coordinates to Failed coordinates array
        fail_coords["x"] = np.append(fail_coords["x"], x_val)
        fail_coords["y"] = np.append(fail_coords["y"], y_val)
        fail_coords["orientation"] = np.append(fail_coords["orientation"], orientation)

    ret = {"success_coords": success_coords, "fail_coords": fail_coords}
    return ret


# Coordinate Saving Step 1: Filter heatmap coordinates by success/fail, orientation
def filter_heatmap_coords(success_coords, fail_coords, episode_num, saving_dir):
    """ Save heatmap coordinates split by success and orientation """

    success_normal_idxs = [i for i, x in enumerate(success_coords["orientation"]) if x == "normal"]
    success_rotated_idxs = [i for i, x in enumerate(success_coords["orientation"]) if x == "rotated"]
    success_top_idxs = [i for i, x in enumerate(success_coords["orientation"]) if x == "top"]

    fail_normal_idxs = [i for i, x in enumerate(fail_coords["orientation"]) if x == "normal"]
    fail_rotated_idxs = [i for i, x in enumerate(fail_coords["orientation"]) if x == "rotated"]
    fail_top_idxs = [i for i, x in enumerate(fail_coords["orientation"]) if x == "top"]

    success_coords_idxs = [success_normal_idxs, success_rotated_idxs, success_top_idxs]
    fail_coords_idxs = [fail_normal_idxs, fail_rotated_idxs, fail_top_idxs]

    for index_list in success_coords_idxs:
        if len(index_list) > 0:
            coords_dict_to_array(success_coords, index_list, "/success", episode_num, saving_dir)
            coords_dict_to_array(success_coords, index_list, "/total", episode_num, saving_dir)

    for index_list in fail_coords_idxs:
        if len(index_list) > 0:
            coords_dict_to_array(fail_coords, index_list, "/fail", episode_num, saving_dir)
            coords_dict_to_array(fail_coords, index_list, "/total", episode_num, saving_dir)

    print("Writing to heatmap info file...")
    f = open(saving_dir + "/heatmap_info.txt", "w")
    total_text = "Heatmap Coords \nSaved at: "+saving_dir+"\n\nTotal # Success: "+str(len(success_coords["x"]))+"\nTotal # Fail: "+str(len(fail_coords["x"]))+"\n"
    normal_text = "\nNormal Orientation\n# Success: "+str(len(success_normal_idxs))+"\n# Fail: "+str(len(fail_normal_idxs))+"\n"
    rotated_text = "\nRotated Orientation\n# Success: " + str(len(success_rotated_idxs)) + "\n# Fail: " + str(len(fail_rotated_idxs)) + "\n"
    top_text = "\nTop Orientation\n# Success: " + str(len(success_top_idxs)) + "\n# Fail: " + str(len(fail_top_idxs)) + "\n"
    text = total_text + normal_text + rotated_text + top_text
    f.write(text)
    f.close()


# Coordinate Saving Step 2: Convert coordinate dictionaries to numpy arrays for saving
def coords_dict_to_array(coords_dict, indexes, filename, episode_num, saving_dir):
    """ Convert coordinate dictionary to numpy array for saving """
    coords_x = np.array([coords_dict["x"][i] for i in indexes])
    coords_y = np.array([coords_dict["y"][i] for i in indexes])
    idx = indexes[0]
    orientation = coords_dict["orientation"][idx]

    save_coordinates(coords_x, coords_y, saving_dir+"/"+orientation, filename, episode_num)


# Coordinate Saving Step 3: Save heatmap coordinates by orientation type
def save_coordinates(x,y,file_path,filename,episode_num):
    """ Save heatmap initial object position x,y coordinates
    x: initial object x-coordinate
    y: initial object y-coordinate
    filename: Location to save heatmap coordinates to
    """
    # Ensure file path is created if it doesn't exist
    coord_save_path = Path(file_path)
    coord_save_path.mkdir(parents=True, exist_ok=True)

    ep_str = ""
    if episode_num is not None:
        ep_str = "_"+str(episode_num)

    np.save(file_path + "/" + filename+"_x"+ep_str, x)
    np.save(file_path+ "/" + filename+"_y"+ep_str, y)