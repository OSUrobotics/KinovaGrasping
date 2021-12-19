import gym
import csv
import os,sys
import numpy as np
import mujoco_py
from main_DDPGfD import get_hand_object_coords_dict
# Import plotting code from other directory
plot_path = os.getcwd() + "/plotting_code"
sys.path.insert(1, plot_path)
from heatmap_plot import heatmap_actual_coords
from heatmap_coords import sort_and_save_heatmap_coords


def create_coord_filepaths(env, mode, with_noise, shape_keys):
    """
    Create output directories.
    """
    if with_noise is False:
        noise_str = "no_noise/"
    else:
        noise_str = "with_noise/"
    bad_coordinate = False

    # Make the new VALID valid coordinates filepath
    valid_all_coords_file = "./gym_kinova_gripper/envs/kinova_description/valid_obj_hand_coords/"
    env.create_paths([valid_all_coords_file, valid_all_coords_file + noise_str,
                      valid_all_coords_file + noise_str + str(mode) + "_coords/",
                      valid_all_coords_file + noise_str + str(mode) + "_coords/" + str(env.orientation)])
    valid_all_coords_filepath = valid_all_coords_file + noise_str + str(mode) + "_coords/" + str(env.orientation) + "/"

    # BAD coordinates filepath
    bad_all_coords_file = "./gym_kinova_gripper/envs/kinova_description/bad_obj_hand_coords/"
    env.create_paths([bad_all_coords_file, bad_all_coords_file + noise_str,
                      bad_all_coords_file + noise_str + str(mode) + "_coords/",
                      bad_all_coords_file + noise_str + str(mode) + "_coords/" + str(env.orientation)])
    bad_all_coords_filepath = bad_all_coords_file + noise_str + str(mode) + "_coords/" + str(env.orientation) + "/"

    for shape_name in shape_keys:
        env.create_paths([valid_all_coords_filepath + shape_name])
        env.create_paths([bad_all_coords_filepath+shape_name])

    return valid_all_coords_filepath, bad_all_coords_filepath


def check_collison_for_all_geoms(env):
    """Check each geom in the simulation to see if there are any collisions due to points of contact between two geoms.
    geoms are any of the meshes rendered within the simulation (ex: Object, fingers, etc.). The contact between the
    ground and the object is NOT considered a collision.
    """
    env.update_sim_data()
    num_contacts = env.contacts
    print('number of contacts', num_contacts)
    if num_contacts == 0:
        # No collision between the geoms (hand/ground/object)
        return False
    elif num_contacts == 1:
        contact_geom1_name = env.model.geom_id2name(env.contact_arr[0].geom1)
        contact_geom2_name = env.model.geom_id2name(env.contact_arr[0].geom2) #object_top
        object_names = ["object","object_top","object_bottom"]
        if (contact_geom1_name == "ground" and any(contact_geom2_name == object for object in object_names)) or (any(contact_geom1_name == object for object in object_names) and contact_geom2_name == "ground"):
            print("We have already checked the object for collision -- all good! No collision")
        return False

    for i in range(env.contacts):
        # Note that the contact array has more than `ncon` entries,
        # so be careful to only read the valid entries.
        contact = env.contact_arr[i]
        print('contact:', i)
        print('distance: {:.9f}'.format(contact.dist))
        print('geom1:', contact.geom1, env.model.geom_id2name(contact.geom1))
        print('geom2:', contact.geom2, env.model.geom_id2name(contact.geom2))
        print('contact position:', contact.pos)

        # Use internal functions to read out mj_contactForce
        c_array = np.zeros(6, dtype=np.float64)
        mujoco_py.functions.mj_contactForce(env.model, env.data, i, c_array)

        # Convert the contact force from contact frame to world frame
        ref = np.reshape(contact.frame, (3, 3))
        c_force = np.dot(np.linalg.inv(ref), c_array[0:3])
        c_torque = np.dot(np.linalg.inv(ref), c_array[3:6])
        print('contact force in world frame:', c_force)
        print('contact torque in world frame:', c_torque)
        print()

    # Contacts exist; there is a collision (contact) between the geoms
    return True


def check_for_collision(env,requested_shape,hand_orientation,with_grasp,mode,curr_orient_idx,with_noise,render_coord,obj_coords=None,hand_rotation=None,adjust_coords=None,adjusted_coord_check=0,coord_difficulty=None):
    """ Checks for collision with the hand and the object """
    has_collision = False

    if obj_coords is None:
        obj_x, obj_y, obj_z, hand_x, hand_y, hand_z, _, coords_file = env.determine_obj_hand_coords(requested_shape[0], mode, orient_idx=curr_orient_idx, with_noise=with_noise, coord_difficulty=coord_difficulty)
        obj_coords = [obj_x, obj_y, obj_z]
        hand_rotation = [hand_x, hand_y, hand_z]
        if adjust_coords is not None:
            obj_coords = np.add(obj_coords,adjust_coords["obj_coords_change"])
            hand_rotation = np.add(hand_rotation,adjust_coords["hand_rotation_angle_change"])
        env.set_coords_filename(coords_file)

    # Fill training object list using latin square
    if env.check_obj_file_empty("objects.csv"):
        env.Generate_Latin_Square(5000, "objects.csv", shape_keys=requested_shape)

    state = env.reset(shape_keys=requested_shape, hand_orientation=hand_orientation, with_grasp=with_grasp,
                      env_name="env", mode=mode, orient_idx=curr_orient_idx, with_noise=with_noise, start_pos=obj_coords, hand_rotation=hand_rotation, coord_difficulty=coord_difficulty)

    # Print which file you're running through
    if curr_orient_idx == 0:
        coords_file = env.get_coords_filename()
        print("Coords filename: ", coords_file)

    # Record the hand-object pose within a dict for plotting and saving
    hand_object_coords = get_hand_object_coords_dict(env)
    before_env_obj_coords = env.get_env_obj_coords()
    before_env_hand_coords = env.get_env_hand_coords()

    # Take a step in the simulation (with no action) to see if the object moves based on a collision with the hand
    if render_coord is True:
        done = False
        while not done:
            env.render()
            obs, reward, done, _ = env.step(action=[0, 0, 0, 0])
            after_env_obj_coords = env.get_env_obj_coords()
    else:
        env.step(action=[0, 0, 0, 0])
    # Object cooridnates after conducting a step
    after_env_obj_coords = env.get_env_obj_coords()
    after_env_hand_coords = env.get_env_hand_coords()

    threshold = 0.0001
    object_collision = np.any(abs(before_env_obj_coords - after_env_obj_coords) > threshold)
    # Check if the object is in collision by its movement after placement
    if object_collision:
        print("Object is in collision!!")

        # If the object has moved due to falling, check the new location for collision
        # This allows the simulaiton to adjust itself for up to 5 attempts given the new object location
        if adjusted_coord_check == 5:
            print("A collision still exists for the object!! Bad coordinate...")
            print("Difference: ",abs(before_env_obj_coords - after_env_obj_coords))
            has_collision = True
        else:
            # Check for collision after the object location adjustment
            has_collision, hand_object_coords = check_for_collision(env, requested_shape, hand_orientation, with_grasp, mode, curr_orient_idx, with_noise, render_coord,obj_coords=after_env_obj_coords.tolist(),hand_rotation=hand_rotation,adjusted_coord_check=adjusted_coord_check+1,coord_difficulty=coord_difficulty)
    # Check for collision in the hand as well
    elif np.any(abs(before_env_hand_coords - after_env_hand_coords) > threshold):
        print("Hand has moved or is in collision!!")
        has_collision = check_collison_for_all_geoms(env)

        """
        # Optionally render the object-hand pose to understand the collision
        if has_collision is True:
            print("Hand has a collision!!")
            done = False
            while not done:
                env.render()
                obs, reward, done, _ = env.step(action=[0, 0, 0, 0])
                after_env_hand_coords = env.get_env_hand_coords()
                print("after_env_hand_coords: ", after_env_hand_coords)
        """

    return has_collision, hand_object_coords

def check_within_range(obj_coords, local_obj_coords,x_min,x_max,y_min):
    """ Checks if the coordinate is within the range (minimum/maximum x and y coordinate ranges)
    Determines using the global representation of the object coordinates (obj_coords)
    """
    # Check if the coordinate is within range for the global frame (x,y) limits)
    # Also check the x-axis of the local frame as there are some outlier positions due to the rotation of the hand
    if (x_min <= obj_coords[0] <= x_max and obj_coords[1] >= y_min) and (x_min <= local_obj_coords[0] <= x_max):
        return True
    else:
        return False


def get_coords_from_file(coords_filename):
    """Get the hand-pose coordinates from the desired file and append them to an array.
    Returns an array of object coordinate positions (x,y,z)"""
    data = []
    global_valid_x = []
    global_valid_y = []
    with open(coords_filename) as csvfile:
        checker = csvfile.readline()
        if ',' in checker:
            delim = ','
        else:
            delim = ' '
    # Go back to the top of the file after checking for the delimiter
    with open(coords_filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=delim)
        for coord in reader:
                # Object x, y, z
                data.append([float(coord[0]), float(coord[1]), float(coord[2])])
                global_valid_x.append(float(coord[0]))
                global_valid_y.append(float(coord[1]))
    return data, global_valid_x, global_valid_y


def plot_coords(coords_filename, fig_name, saving_dir, plot_title=""):
    """ Plot coordinates within a certain range -- This plots coordinates that are already saved"""
    x_min = -0.11
    x_max = 0.11
    y_min = -0.11
    y_max = 0.11

    data, global_valid_x, global_valid_y = get_coords_from_file(coords_filename)

    heatmap_actual_coords(total_x=global_valid_x, total_y=global_valid_y,
                          plot_title="Global Frame: Object coordinates " + plot_title,
                          fig_filename=fig_name+' Coords_global_actual_heatmap.png', saving_dir=saving_dir,
                          hand_lines=None, state_rep="global",x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max)


def plot_coords_for_each_shape(shape_keys,with_noise,hand_orientation):
    """
    Plot the object-hand coordinates for each shape.
    """
    if with_noise is False:
        noise_str = "no_noise/"
    else:
        noise_str = "with_noise/"

    for shape in shape_keys:
        coords_dir = "./gym_kinova_gripper/envs/kinova_description/valid_obj_hand_coords_new/" + noise_str + "shape_coords/" + hand_orientation + "/"
        coords_filename = coords_dir + shape + ".txt"
        fig_name = shape

        print("Plotting: ",coords_filename)
        plot_coords(coords_filename, fig_name, coords_dir, plot_title=fig_name)


def create_hand_object_plots(coords, shape, coord_type, saving_dir, use_text_file_coords=False):
        """
        Generate plots displaying the actual coordinate location of each object-hand position
        """
        actual_plot_title = "Initial Coordinate Position of the Object\n"
        x_min = -0.11
        x_max = 0.11
        y_min = -0.11
        y_max = 0.11

        if use_text_file_coords is False:
            for frame in ["local_",""]:
                x_vals = [d[frame + "obj_coords"][0] for d in coords]
                y_vals = [d[frame + "obj_coords"][1] for d in coords]
                plot_title = frame + " frame: " + coord_type +" "+ actual_plot_title
                fig_filename = shape + "_" + coord_type + "_coords_"+frame+"_actual_heatmap.png"
                heatmap_actual_coords(total_x=x_vals, total_y=y_vals,
                                      plot_title=plot_title,
                                      fig_filename=fig_filename,
                                      saving_dir=saving_dir, hand_lines=None, state_rep=frame, x_min=x_min,
                                      x_max=x_max, y_min=y_min, y_max=y_max)
            else:
                x_vals = [d[0] for d in coords]
                y_vals = [d[1] for d in coords]
                plot_title = coord_type +" "+ actual_plot_title
                fig_filename = shape + "_" + coord_type + "_coords_"+frame+"_actual_heatmap.png"
                heatmap_actual_coords(total_x=x_vals, total_y=y_vals,
                                      plot_title=plot_title,
                                      fig_filename=fig_filename,
                                      saving_dir=saving_dir, hand_lines=None, state_rep=frame, x_min=x_min,
                                      x_max=x_max, y_min=y_min, y_max=y_max)


def coord_check_loop(shape_keys, with_noise, hand_orientation, render_coord, adjust_coords=None, orient_idx=None, coord_difficulty=None):
    """Loop through each coordinate within the all_shapes file based on the given shape, hand orientation amd
       hand orientation variation.
    """
    # Input needed to determine the object-hand coordinates: random_shape, mode, orient_idx=orient_idx, with_noise=with_noise
    env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    with_grasp = False
    mode = "shape"
    env.set_orientation(hand_orientation)
    if coord_difficulty is None:
        coord_difficulty_str = ""
    else:
        coord_difficulty_str = "_" + coord_difficulty
    
    valid_all_coords_filepath, bad_all_coords_filepath = create_coord_filepaths(env,mode,with_noise,shape_keys)

    for shape_name in shape_keys:
        valid_data = []
        bad_data =[]
        valid_shape_coords_filepath = valid_all_coords_filepath + shape_name + coord_difficulty_str
        bad_shape_coords_filepath = bad_all_coords_filepath + shape_name + coord_difficulty_str
        requested_shape = [shape_name]

        # Generate randomized list of objects to select from
        env.Generate_Latin_Square(5000, "objects.csv", shape_keys=requested_shape)

        valid_hand_object_coords = []
        bad_hand_object_coords = []

        # If no desired file index is selected, loop through whole file
        if orient_idx is None:
            _, _, _, _, _, _, _, coords_file = env.determine_obj_hand_coords(random_shape=shape_name, mode="shape", orient_idx=0, with_noise=with_noise, coord_difficulty=coord_difficulty)
            num_lines = 0
            with open(coords_file) as f:
                for line in f:
                    num_lines = num_lines + 1
            indexes = range(num_lines)
        else:
            indexes = [orient_idx]

        # Check each coordinate within the file
        for curr_orient_idx in indexes:
            print("Coord File Idx: ", curr_orient_idx)

            # Returns True is there is a collision
            has_collision, hand_object_coords = check_for_collision(env,requested_shape,hand_orientation,with_grasp,mode,curr_orient_idx,with_noise,render_coord,adjust_coords=adjust_coords,coord_difficulty=coord_difficulty)

            # Returns True if the coordinate is within the min and max x,y coordinate ranges
            within_range = True
            if hand_orientation == "normal":
                x_min = -0.09
                x_max = 0.09
                y_min = -0.01
            elif hand_orientation == "rotated":
                x_min = -0.09
                x_max = 0.09
                y_min = -0.06
            elif hand_orientation == "top":
                x_min = -0.10
                x_max = 0.10
                y_min = -0.04

            within_range = check_within_range(hand_object_coords["obj_coords"],hand_object_coords["local_obj_coords"],x_min,x_max,y_min)

            # Check if the coordinate has moved after conducting an action
            if has_collision is False and within_range is True:
                #print("Coordinate is VALID!! Writing it to file: object x,y,z: {}, hov: {}".format(obj_coords,hand_orient_variation))

                # Add coordinated to valid (x,y) coordinate list for plotting
                valid_hand_object_coords.append(hand_object_coords)
                
                # Append coordinate
                if with_noise is True:
                    # Object x, y, z coordinates, followed by corresponding hand orientation x, y, z coords
                    valid_data.append(
                        [float(hand_object_coords["obj_coords"][0]), float(hand_object_coords["obj_coords"][1]), float(hand_object_coords["obj_coords"][2]),
                         float(hand_object_coords["hov"][0]), float(hand_object_coords["hov"][1]),
                         float(hand_object_coords["hov"][2])])
                else:
                    # No hov coordinates are just the object coordinates (no change in the default hand orientation)
                    valid_data.append([float(hand_object_coords["obj_coords"][0]), float(hand_object_coords["obj_coords"][1]), float(hand_object_coords["obj_coords"][2])])
            else:
                # if its not close: discard that datapoint, don't use it/ delete it/mark it.
                print("Invalid! object x,y,z: object x,y,z: {}, hov: {}".format(hand_object_coords["obj_coords"], hand_object_coords["hov"]))

                # Add coordinated to bad (x,y) coordinate list for plotting
                bad_hand_object_coords.append(hand_object_coords)
                
                # Append coordinate
                if with_noise is True:
                    # Object x, y, z coordinates, followed by corresponding hand orientation x, y, z coords
                    bad_data.append(
                        [float(hand_object_coords["obj_coords"][0]), float(hand_object_coords["obj_coords"][1]), float(hand_object_coords["obj_coords"][2]),
                         float(hand_object_coords["hov"][0]), float(hand_object_coords["hov"][1]),
                         float(hand_object_coords["hov"][2])])
                else:
                    # Only object coordinates
                    bad_data.append([float(hand_object_coords["obj_coords"][0]), float(hand_object_coords["obj_coords"][1]), float(hand_object_coords["obj_coords"][2])])

        print("Writing coordinates to: ", valid_shape_coords_filepath + ".txt")
        # Write VALID valid coordinate data to file
        with open(valid_shape_coords_filepath + ".txt", 'w', newline='') as outfile:
            w_shapes = csv.writer(outfile, delimiter=' ')
            for coord_values in valid_data:
                w_shapes.writerow(coord_values)

        print("Writing coordinates to: ", bad_shape_coords_filepath + ".txt")
        # Write BAD coordinate data to file
        with open(bad_shape_coords_filepath + ".txt", 'w', newline='') as outfile:
            w_shapes = csv.writer(outfile, delimiter=' ')
            for coord_values in bad_data:
                w_shapes.writerow(coord_values)

        if len(valid_hand_object_coords) > 0:
            dict_file = open(valid_shape_coords_filepath + "/" + shape_name + "_valid_hand_object_coords.csv", "w", newline='')
            keys = valid_hand_object_coords[0].keys()
            dict_writer = csv.DictWriter(dict_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(valid_hand_object_coords)
            dict_file.close()

        if len(bad_hand_object_coords) > 0:
            dict_file = open(bad_shape_coords_filepath + "/" + shape_name + "_bad_hand_object_coords.csv", "w", newline='')
            keys = bad_hand_object_coords[0].keys()
            dict_writer = csv.DictWriter(dict_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(bad_hand_object_coords)
            dict_file.close()

        ## Generate heatmap plots ###
        all_hand_object_coords = valid_hand_object_coords + bad_hand_object_coords

        print("Generating heatmap plots for the object coordinates in the local and global frame....")
        create_hand_object_plots(valid_hand_object_coords, shape_name, "valid", valid_shape_coords_filepath + "/")
        create_hand_object_plots(bad_hand_object_coords, shape_name, "bad", bad_shape_coords_filepath + "/")
        create_hand_object_plots(all_hand_object_coords, shape_name, "all", valid_shape_coords_filepath + "/")


def read_text_file_coords(with_noise,coords_file):
    """
    Read coordinates from the text file
    """
    data = []
    with open(coords_file) as csvfile:
        checker = csvfile.readline()
        if ',' in checker:
            delim = ','
        else:
            delim = ' '
    # Go back to the top of the file after checking for the delimiter
    with open(coords_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=delim)
        for i in reader:
            if with_noise is True:
                # Object x, y, z coordinates, followed by corresponding hand orientation x, y, z coords
                data.append([float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4]), float(i[5])])
            else:
                # Hand orientation is set to (0, 0, 0) if no orientation is selected
                data.append([float(i[0]), float(i[1]), float(i[2]), 0, 0, 0])

    return data

def remove_coord_outliers(x_min,x_max,y_min,y_max,with_noise,coords_dir,filename):
    """
    After coordinates have been checked for collision, remove any coordinates that lie outside of a specific range
    """
    # Read coordinates from the text file and csv file
    all_coords = read_text_file_coords(with_noise, coords_dir+filename)

    # Check if coordinates are within range
    valid_coords = []
    bad_coords = []
    orient_indexes = []
    bad_indexes = []
    for i in range(len(all_coords)):
        if x_min <= all_coords[i][0] <= x_max and y_min <= all_coords[i][1] <= y_max:
            valid_coords.append(all_coords[i])
            orient_indexes.append(i)
        else:
            bad_coords.append(all_coords[i])
            bad_indexes.append(i)

    create_hand_object_plots(valid_hand_object_coords, shape_name, "valid", coords_dir)
    print("done")
    # Write coordinates that are within range

    # Plot newly-filtered coordinates
    #create_hand_object_plots(valid_hand_object_coords, shape_name, "valid", valid_shape_coords_filepath + "/")

if __name__ == "__main__":
    all_shapes = ["CubeB","CubeM","CubeB","CylinderM","Vase1M"] # ["CylinderB","Cube45S","Cube45B","Cone1S","Cone1B","Cone2S","Cone2B","Vase1S","Vase1B","Vase2S","Vase2B"]

    adjust_coords = {}
    adjust_coords["obj_coords_change"] = [0,0,0]
    adjust_coords["hand_rotation_angle_change"] = [0,0,0]#[-0.15,0,0] #[-0.0872665,0,0]
    difficulty = None
    filtering_type = "remove_outliers"

    if filtering_type == "collision":
        for shape in all_shapes:
            shape_keys = [shape]
            print("*** Filtering ",shape_keys)
            coord_check_loop(shape_keys=shape_keys, with_noise=True, orient_idx=None, hand_orientation="normal", adjust_coords=adjust_coords, render_coord=False, coord_difficulty=difficulty)
    elif filtering_type == "remove_outliers":
        x_min = -0.09
        x_max = 0.09
        y_min = -0.06
        y_max = 0.02
        with_noise = True
        orientation = "rotated"
        shapes = []

        if with_noise is True:
            noise_str = "with_noise/"
        else:
            noise_str = "no_noise/"

        for shape in all_shapes:
            coords_dir = "gym_kinova_gripper/envs/kinova_description/valid_obj_hand_coords/" + noise_str + "shape_coords/" + orientation + "/"
            remove_coord_outliers(x_min,x_max,y_min,y_max,with_noise,coords_dir,filename=shape+".txt")

    #print("Done looping through coords - Quitting")
    #plot_coords_for_each_shape(shape_keys,with_noise=True,hand_orientation="normal")