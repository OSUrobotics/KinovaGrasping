import gym
import csv
import os,sys
import numpy as np
import mujoco_py
# Import plotting code from other directory
plot_path = os.getcwd() + "/plotting_code"
sys.path.insert(1, plot_path)
from heatmap_plot import heatmap_actual_coords

def create_coord_filepaths(env, mode, with_noise):
    # coords filename structure: "gym_kinova_gripper/envs/kinova_description/obj_hand_coords/" + noise_file + str(mode)+"_coords/" + str(env.orientation) + "/" + random_shape + ".txt"
    if with_noise is False:
        noise_str = "no_noise/"
    else:
        noise_str = "with_noise/"
    bad_coordinate = False

    # Make the new VALID valid coordinates filepath
    valid_all_coords_file = "./gym_kinova_gripper/envs/kinova_description/valid_obj_hand_coords_new/"
    env.create_paths([valid_all_coords_file, valid_all_coords_file + noise_str,
                      valid_all_coords_file + noise_str + str(mode) + "_coords/",
                      valid_all_coords_file + noise_str + str(mode) + "_coords/" + str(env.orientation)])
    valid_all_coords_filepath = valid_all_coords_file + noise_str + str(mode) + "_coords/" + str(env.orientation)

    # BAD coordinates filepath
    bad_all_coords_file = "./gym_kinova_gripper/envs/kinova_description/bad_obj_hand_coords_new/"
    env.create_paths([bad_all_coords_file, bad_all_coords_file + noise_str,
                      bad_all_coords_file + noise_str + str(mode) + "_coords/",
                      bad_all_coords_file + noise_str + str(mode) + "_coords/" + str(env.orientation)])
    bad_all_coords_filepath = bad_all_coords_file + noise_str + str(mode) + "_coords/" + str(env.orientation)

    return valid_all_coords_filepath, bad_all_coords_filepath

def check_collison_for_all_geoms(env):
    env.update_sim_data()
    num_contacts = env.contacts
    print('number of contacts', num_contacts)
    if num_contacts == 0:
        # No collision between the geoms (hand/ground/object)
        return False
    elif num_contacts == 1:
        contact_geom1_name = env.model.geom_id2name(env.contact_arr[0].geom1)
        contact_geom2_name = env.model.geom_id2name(env.contact_arr[0].geom2)
        if (contact_geom1_name == "ground" and contact_geom2_name == "object") or (contact_geom1_name == "object" and contact_geom2_name == "ground"):
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


def check_for_collision(env,requested_shape,hand_orientation,with_grasp,mode,curr_orient_idx,with_noise,render_coord,obj_coords=None,hand_rotation=None,adjusted_coord_check=0):
    """ Checks for collision with the hand and the object """
    has_collision = False

    if obj_coords is None:
        obj_x, obj_y, obj_z, hand_x, hand_y, hand_z, _, coords_file = env.determine_obj_hand_coords(requested_shape[0], mode, orient_idx=curr_orient_idx, with_noise=with_noise)
        obj_coords = [obj_x, obj_y, obj_z]
        hand_rotation = [hand_x, hand_y, hand_z]
        env.set_coords_filename(coords_file)

    # Fill training object list using latin square
    if env.check_obj_file_empty("objects.csv"):
        env.Generate_Latin_Square(file_size, "objects.csv", shape_keys=requested_shape)

    state = env.reset(shape_keys=requested_shape, hand_orientation=hand_orientation, with_grasp=with_grasp,
                      env_name="env", mode=mode, orient_idx=curr_orient_idx, with_noise=with_noise, start_pos=obj_coords, hand_rotation=hand_rotation)

    # Print which file you're running through
    if curr_orient_idx == 0:
        coords_file = env.get_coords_filename()
        print("Coords filename: ", coords_file)

    # Get the current object coordinate and hand orientation pair
    obj_coords = env.get_obj_coords()
    local_state = env.get_obs_from_coord_frame(coord_frame="local")
    local_obj_coords = local_state[21:24]

    before_env_obj_coords = env.get_env_obj_coords()
    #print("before_env_obj_coords: ",before_env_obj_coords)
    before_env_hand_coords = env.get_env_hand_coords()
    hand_orient_variation = env.hand_orient_variation

    # Take a step in the simulation (with no action) to see if the object moves based on a collision with the hand
    if render_coord is True:
        done = False
        while not done:
            env.render()
            obs, reward, done, _ = env.step(action=[0, 0, 0, 0])
            after_env_obj_coords = env.get_env_obj_coords()
            #print("after_env_obj_coords: ",after_env_obj_coords)
    else:
        env.step(action=[0, 0, 0, 0])

        # Object cooridnates after conducting a step
        after_env_obj_coords = env.get_env_obj_coords()
        after_env_hand_coords = env.get_env_hand_coords()

    threshold = 0.0001
    if np.any(abs(before_env_obj_coords - after_env_obj_coords) > threshold):
        #print("Object is in collision!!")
        #print("Object moved -- Trying adjustment...")
        if adjusted_coord_check == 5:
            print("A collision still exists for the object!! Bad coordinate...")
            print("Difference: ",abs(before_env_obj_coords - after_env_obj_coords))
            has_collision = True
        else:
            has_collision, local_obj_coords, obj_coords, hand_orient_variation = check_for_collision(env, requested_shape, hand_orientation, with_grasp, mode, curr_orient_idx, with_noise, render_coord,obj_coords=after_env_obj_coords.tolist(),hand_rotation=hand_rotation,adjusted_coord_check=adjusted_coord_check+1)
            #print("After adjustment!! obj_coords: ",obj_coords)
    elif np.any(abs(before_env_hand_coords - after_env_hand_coords) > threshold):
        #print("Hand is in has moved or is in collision!!")
        #print("Checking which geoms are in collision!! obj_coords: {}, hov: {}".format(obj_coords, hand_orient_variation))
        has_collision = check_collison_for_all_geoms(env)

        if has_collision is True:
            print("Hand has a collision!!")
            """
            done = False
            while not done:
                env.render()
                obs, reward, done, _ = env.step(action=[0, 0, 0, 0])
                after_env_hand_coords = env.get_env_hand_coords()
                print("after_env_hand_coords: ", after_env_hand_coords)
            """

    return has_collision, local_obj_coords, obj_coords, hand_orient_variation

def check_within_range(obj_coords):
    """ Checks if the coordinate is within the range (minimum/maximum x and y coordinate ranges)
    Determines using the global representation of the object coordinates (obj_coords)
    """
    x_min = -0.09
    x_max = 0.09
    y_min = -0.01

    # check if the object coord is within range
    if x_min <= obj_coords[0] <= x_max and obj_coords[1] >= y_min:
        return True
    else:
        return False

def coord_check_loop(shape_keys, with_noise, hand_orientation, render_coord, file_size, orient_idx=None):
    """Loop through each coordinate within the all_shapes file based on the given shape, hand orientation amd
       hand orientation variation.
    """
    # Input needed to determine the object-hand coordinates: random_shape, mode, orient_idx=orient_idx, with_noise=with_noise
    env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    with_grasp = False
    mode = "shape"
    
    valid_all_coords_filepath, bad_all_coords_filepath = create_coord_filepaths(env,mode,with_noise)

    for shape_name in shape_keys:
        valid_data = []
        bad_data =[]
        valid_shape_coords_file = valid_all_coords_filepath + "/" + shape_name + ".txt"
        bad_shape_coords_file = bad_all_coords_filepath + "/" + shape_name + ".txt"
        requested_shape = [shape_name]

        # Generate randomized list of objects to select from
        env.Generate_Latin_Square(file_size, "objects.csv", shape_keys=requested_shape)

        # valid coordinates for plotting
        local_valid_x = []
        local_valid_y = []
        global_valid_x = []
        global_valid_y = []

        # Bad coordinates for plotting
        local_bad_x = []
        local_bad_y = []
        global_bad_x = []
        global_bad_y = []

        # If no desired file index is selected, loop through whole file
        if orient_idx is None:
            indexes = range(file_size)
        else:
            indexes = [orient_idx]

        # Check each coordinate within the file
        for curr_orient_idx in indexes:
            print("Coord File Idx: ", curr_orient_idx)

            # Returns True is there is a collision
            has_collision, local_obj_coords, obj_coords, hand_orient_variation = check_for_collision(env,requested_shape,hand_orientation,with_grasp,mode,curr_orient_idx,with_noise,render_coord)

            # Returns True if the coordinate is within the min and max x,y coordinate ranges
            within_range = check_within_range(obj_coords)

            # Check if the coordinate has moved after conducting an action
            if has_collision is False and within_range is True:
                #print("Coordinate is VALID!! Writing it to file: object x,y,z: {}, hov: {}".format(obj_coords,hand_orient_variation))

                # Add coordinated to valid (x,y) coordinate list for plotting
                local_valid_x.append(local_obj_coords[0])
                local_valid_y.append(local_obj_coords[1])
                global_valid_x.append(obj_coords[0])
                global_valid_y.append(obj_coords[1])
                
                # Append coordinate
                if with_noise is True:
                    # Object x, y, z coordinates, followed by corresponding hand orientation x, y, z coords
                    valid_data.append(
                        [float(obj_coords[0]), float(obj_coords[1]), float(obj_coords[2]),
                         float(hand_orient_variation[0]), float(hand_orient_variation[1]),
                         float(hand_orient_variation[2])])
                else:
                    # No hov coordinates are just the object coordinates (no change in the default hand orientation)
                    valid_data.append([float(obj_coords[0]), float(obj_coords[1]), float(obj_coords[2])])
            else:
                # if its not close: discard that datapoint, don't use it/ delete it/mark it.
                print("Invalid! object x,y,z: object x,y,z: {}, hov: {}".format(obj_coords, hand_orient_variation))

                # Add coordinated to bad (x,y) coordinate list for plotting
                local_bad_x.append(local_obj_coords[0])
                local_bad_y.append(local_obj_coords[1])
                global_bad_x.append(obj_coords[0])
                global_bad_y.append(obj_coords[1])
                
                # Append coordinate
                if with_noise is True:
                    # Object x, y, z coordinates, followed by corresponding hand orientation x, y, z coords
                    bad_data.append(
                        [float(obj_coords[0]), float(obj_coords[1]), float(obj_coords[2]),
                         float(hand_orient_variation[0]), float(hand_orient_variation[1]),
                         float(hand_orient_variation[2])])
                else:
                    # Hand orientation is set to (0, 0, 0) if no orientation is selected
                    bad_data.append([float(obj_coords[0]), float(obj_coords[1]), float(obj_coords[2])])

        print("Writing coordinates to: ", valid_shape_coords_file)
        # Write VALID valid coordinate data to file
        with open(valid_shape_coords_file, 'w', newline='') as outfile:
            w_shapes = csv.writer(outfile, delimiter=' ')
            for coord_values in valid_data:
                w_shapes.writerow(coord_values)

        print("Writing coordinates to: ", bad_shape_coords_file)
        # Write BAD coordinate data to file
        with open(bad_shape_coords_file, 'w', newline='') as outfile:
            w_shapes = csv.writer(outfile, delimiter=' ')
            for coord_values in bad_data:
                w_shapes.writerow(coord_values)

        print("Generating heatmap plots for the object coordinates in the local and global frame....")
        actual_plot_title = "Initial Coordinate Position of the Object\n"
        # Valid coordinates
        heatmap_actual_coords(total_x=local_valid_x, total_y=local_valid_y, plot_title="Local Frame: Valid "+actual_plot_title, fig_filename='valid_coords_local_actual_heatmap.png', saving_dir=valid_shape_coords_file, hand_lines=None, state_rep="local")
        heatmap_actual_coords(total_x=global_valid_x, total_y=global_valid_y, plot_title="Global Frame: Valid "+actual_plot_title, fig_filename='valid_coords_global_actual_heatmap.png', saving_dir=valid_shape_coords_file, hand_lines=None, state_rep="global")

        # Bad coordinates
        heatmap_actual_coords(total_x=local_bad_x, total_y=local_bad_y, plot_title="Local Frame: Bad"+actual_plot_title, fig_filename='bad_coords_local_actual_heatmap.png', saving_dir=bad_shape_coords_file, hand_lines=None, state_rep="local")
        heatmap_actual_coords(total_x=global_bad_x, total_y=global_bad_y, plot_title="Global Frame: Bad"+actual_plot_title, fig_filename='bad_coords_global_actual_heatmap.png', saving_dir=bad_shape_coords_file, hand_lines=None, state_rep="global")

        # All coordinates
        all_coords_local_x = np.append(local_valid_x,local_bad_x)
        all_coords_local_y = np.append(local_valid_y, local_bad_y)
        all_coords_global_x = np.append(global_valid_x, global_bad_x)
        all_coords_global_y = np.append(global_valid_y, global_bad_y)

        heatmap_actual_coords(total_x=all_coords_local_x, total_y=all_coords_local_y,
                              plot_title="Local Frame: ALL " + actual_plot_title,
                              fig_filename='all_coords_local_actual_heatmap.png', saving_dir=valid_shape_coords_file,
                              hand_lines=None, state_rep="local")
        heatmap_actual_coords(total_x=all_coords_global_x, total_y=all_coords_global_y,
                              plot_title="Global Frame: ALL " + actual_plot_title,
                              fig_filename='all_coords_global_actual_heatmap.png', saving_dir=valid_shape_coords_file,
                              hand_lines=None, state_rep="global")


if __name__ == "__main__":
    shape_keys = ["CubeM","CubeS","CubeB","CylinderM","Vase1M"]  # ["CylinderB","Cube45S","Cube45B","Cone1S","Cone1B","Cone2S","Cone2B","Vase1S","Vase1B","Vase2S","Vase2B"]
    file_size = 5000
    coord_check_loop(shape_keys=shape_keys, with_noise=False, orient_idx=None, hand_orientation="normal", file_size=file_size, render_coord=False)
    print("Done looping through coords - Quitting")