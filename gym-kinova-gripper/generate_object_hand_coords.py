import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import csv

""" Generate object-hand coordinates per shape/size/hand orientation with or without
    Hand Orientation Variation """

def generate_points_in_graspable_region(pt1, pt2, pt3):
    """
    Random point on the triangle with vertices pt1, pt2 and pt3.
    """
    x, y = random.random(), random.random()
    q = abs(x - y)
    s, t, u = q, 0.5 * (x + y - q), 1 - 0.5 * (q + x + y)
    return (
        s * pt1[0] + t * pt2[0] + u * pt3[0],
        s * pt1[1] + t * pt2[1] + u * pt3[1],
    )


def generate_object_coordinates(num_points,shape_name,orient_name):
    """
    Generate the set of initial object coordinates within the graspable range of the hand.

    """
    object_half_heights = {"CubeS": 0.04792,"CubeM": 0.05274,"CubeB": 0.05745,"CylinderM": 0.05498, "Vase1M": 0.05500}
    graspable_regions = {"normal": [-0.09,0.09,-0.01,0.04],"rotated": [-0.06, 0.06,-0.04, 0.03],"top": [-0.06, 0.06,-0.09, 0.09]} # x_min, x_max, y_min, y_max
    x_min, x_max, y_min, y_max = graspable_regions[orient_name][0], graspable_regions[orient_name][1], graspable_regions[orient_name][2], graspable_regions[orient_name][3]

    # Generate x,y coordinates within the graspable region
    if orient_name == "normal":
        pt1 = (x_min, y_min)
        pt2 = (0.0, y_max)
        pt3 = (x_max, y_min)
        x_coords, y_coords = zip(*[generate_points_in_graspable_region(pt1, pt2, pt3) for _ in range(num_points)])
    else:
        pt1 = (x_min, y_min)
        pt2 = (x_min,y_max)
        pt3 = (x_max, y_max)
        pt4 = (x_max, y_min)
        half_num_points = int(num_points/2)

        # Generate points within a polygon (two triangular regions)
        x_coords, y_coords = zip(*[generate_points_in_graspable_region(pt1, pt2, pt3) for _ in range(half_num_points)])
        x_coords_2, y_coords_2 = zip(*[generate_points_in_graspable_region(pt1, pt3, pt4) for _ in range(half_num_points)])
        x_coords = list(x_coords)
        y_coords = list(y_coords)
        x_coords.extend(list(x_coords_2))
        y_coords.extend(list(y_coords_2))

    z = object_half_heights[shape_name]

    # Complete list of object coordinates
    obj_coords = [[x, y, z] for x, y in zip(x_coords, y_coords)]

    # Plot points
    plt.scatter(x_coords, y_coords, s=1)
    plt.show()

    return obj_coords


def generate_hand_rotations(num_points,orient_name):
    """
    Generate a list of wrist coordinates for the robot hand
    """
    hand_rotations = np.zeros([num_points, 3])
    rotation_variation = np.zeros([3])

    orient_euler_angles = {"normal": [-1.57, 0, -1.57], "rotated": [-1.2, 0, 0], "top": [0, 0, 0]}

    for i in range(num_points):
        # Added random variation +/- 5 degrees
        #rotation_variation[1:] = np.random.normal(-0.087, 0.087, 2) # Only add variation to the pitch/yaw
        rotation_variation = np.random.normal(-0.087, 0.087, 3)
        hand_rotations[i, :] = np.array(orient_euler_angles[orient_name]) + rotation_variation

    return hand_rotations


def write_transformations_to_file(obj_hand_poses, orient, shape, with_noise):
    """
    Write the xml transformations for both the hand and the object.
    """
    if with_noise is False:
        noise_str = "no_noise"
    else:
        noise_str = "with_noise"

    file_path = "./gym_kinova_gripper/envs/kinova_description/obj_hand_poses/" + noise_str + "/" + orient + "/" + shape
    print("Writing coordinates to: ", file_path + ".txt")

    # Write object-hand poses to file
    with open(file_path + ".txt", 'w', newline='') as outfile:
        w_shapes = csv.writer(outfile, delimiter=' ')
        for pose in obj_hand_poses:
            w_shapes.writerow(pose)


def render_object_hand_pose(obj_coords=None,wrist_coords=None,requested_shape=["CubeM"],hand_orientation="normal"):
    """
    Render the hand and object in a specific coordinate position
    """
    env = gym.make("gym_kinova_gripper:kinovagripper-v0")

    done = False
    while not done:
        state = env.reset(shape_keys=requested_shape, hand_orientation=hand_orientation, start_pos=obj_coords, hand_rotation=wrist_coords, with_noise=True)
        print("RESET ENVIRONMENT WITH OBJ COORDS: ", obj_coords)
        env.render(setPause=True)
        obs, reward, done, _ = env.step(action=[0, 0, 0, 0])


if __name__ == "__main__":
    num_points = 6000 # Number of HOV points to generate
    shape_names= ["CubeS","CubeM","CubeB", "CylinderM", "Vase1M"]
    orient_names= ["normal","rotated","top"]
    with_noise = False

    for orient in orient_names:
        for shape in shape_names:
            obj_hand_poses = []
            # STEP 1: GENERATE OBJECT POSITIONS FOR EACH HAND ORIENTATION
            # Generate the object's initial coordinate positions (x,y,z)
            obj_coords = generate_object_coordinates(num_points, shape_name=shape,orient_name=orient)

            if with_noise is True:
                # STEP 2: GENERATE HAND ROTATIONS IN RELATION TO EACH OBJECT
                # Generate the robot hand (wrist) initial coordinate positions (x,y,z)
                hand_rotations = generate_hand_rotations(num_points,orient)

                for obj, hand in zip(obj_coords, hand_rotations):
                    pose = np.append(obj, hand)
                    obj_hand_poses.append(pose)

            else:
                obj_hand_poses = obj_coords

            # STEP 3: WRITE EACH FULL SET OF OBJECT-HAND COORDINATES TO FILE
            # Write the object-hand coordinate pairs to a text file
            write_transformations_to_file(obj_hand_poses, orient, shape, with_noise)

    # STEP 4: CHECK OBJECT-HAND COORDINATES FOR COLLISION -- use check_coordinates_for_collision.py
    # STEP 5: SPLIT OBJECT-HAND COORDINATES INTO TRAIN/EVAL/TEST SETS -- use separate_train_test_coords.py

    # Note: For testing purposes, call the following function with a specific object-hand pose
    # render_object_hand_pose(wrist_coords=[0,0,0],requested_shape=shape,hand_orientation=orient)