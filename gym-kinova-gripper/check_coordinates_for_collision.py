import gym
import csv
import os,sys
# Import plotting code from other directory
plot_path = os.getcwd() + "/plotting_code"
sys.path.insert(1, plot_path)
from heatmap_plot import heatmap_actual_coords

def loop_through_coords(shape_keys, with_noise, hand_orientation, file_size):
    """Loop through each coordinate within the all_shapes file based on the given shape, hand orientation amd
       hand orientation variation.
    """
    # Input needed to determine the object-hand coordinates: random_shape, mode, orient_idx=orient_idx, with_noise=with_noise
    env = gym.make('gym_kinova_gripper:kinovagripper-v0')
    
    # coords filename structure: "gym_kinova_gripper/envs/kinova_description/obj_hand_coords/" + noise_file + str(mode)+"_coords/" + str(env.orientation) + "/" + random_shape + ".txt"
    with_grasp = False
    if with_noise is False:
        noise_str = "no_noise/"
    else:
        noise_str = "with_noise/"
    mode = "shape"
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

    for shape_name in shape_keys:
        valid_data = []
        bad_data =[]
        valid_shape_coords_file = valid_all_coords_filepath + "/" + shape_name + ".txt"
        bad_shape_coords_file = bad_all_coords_filepath + "/" + shape_name + ".txt"
        requested_shapes = [shape_name]

        # Generate randomized list of objects to select from
        env.Generate_Latin_Square(file_size, "objects.csv", shape_keys=requested_shapes)

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

        # Check each coordinate within the file
        for orient_idx in range(file_size):
            print("Coord File Idx: ", orient_idx)
            state = env.reset(shape_keys=requested_shapes, hand_orientation=hand_orientation, with_grasp=with_grasp,
                               env_name="env", mode=mode, orient_idx=orient_idx, with_noise=with_noise)

            # Print which file you're running through
            if orient_idx == 0:
                coords_file = env.get_coords_filename()
                print("Coords filename: ", coords_file)

            # Get the current object coordinate and hand orientation pair
            obj_coords = env.get_obj_coords()
            local_state = env.get_obs_from_coord_frame(coord_frame="local")
            local_obj_coords = local_state[21:24]

            before_env_obj_coords = env.get_env_obj_coords()
            hand_orient_variation = env.hand_orient_variation

            # Take a step in the simulation (with no action) to see if the object moves based on a collision with the hand
            env.step(action=[0, 0, 0, 0])

            # Object cooridnates after conducting a step
            after_env_obj_coords = env.get_env_obj_coords() #env._sim.data.get_geom_xpos("object")

            # Check if the coordinate has moved after conducting an action
            if before_env_obj_coords[0] == after_env_obj_coords[0] and before_env_obj_coords[1] == after_env_obj_coords[1] and before_env_obj_coords[2] == after_env_obj_coords[2]:
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
                bad_coordinate = True  # Signal that the current coordinate is bad
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
        heatmap_actual_coords(total_x=local_valid_x, total_y=local_valid_y, plot_title="Local Frame: Valid "+actual_plot_title, fig_filename='valid_coords_local_actual_heatmap.png', saving_dir=valid_all_coords_file, hand_lines=None, state_rep="local")
        heatmap_actual_coords(total_x=global_valid_x, total_y=global_valid_y, plot_title="Global Frame: Valid "+actual_plot_title, fig_filename='valid_coords_global_actual_heatmap.png', saving_dir=valid_all_coords_file, hand_lines=None, state_rep="global")
        
        heatmap_actual_coords(total_x=local_bad_x, total_y=local_bad_y, plot_title="Local Frame: Bad"+actual_plot_title, fig_filename='bad_coords_local_actual_heatmap.png', saving_dir=bad_all_coords_file, hand_lines=None, state_rep="local")
        heatmap_actual_coords(total_x=global_bad_x, total_y=global_bad_y, plot_title="Global Frame: Bad"+actual_plot_title, fig_filename='bad_coords_global_actual_heatmap.png', saving_dir=bad_all_coords_file, hand_lines=None, state_rep="global")


if __name__ == "__main__":
    shape_keys = ["CubeM"] #, "CubeS" ,"CubeB" ,"CylinderM", "Vase2M"]  # ["CylinderB","Cube45S","Cube45B","Cone1S","Cone1B","Cone2S","Cone2B","Vase1S","Vase1B","Vase2S","Vase2B"]
    file_size = 5000
    loop_through_coords(shape_keys=shape_keys, with_noise=True, hand_orientation="normal", file_size=file_size)
    print("Done looping through coords - Quitting")