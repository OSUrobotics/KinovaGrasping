import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import csv

""" Generate object-hand coordinates per shape/size/hand orientation with or without
    Hand Orientation Variation """

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def random_points_within(poly, num_coords):
    """
    Get a randomly-selected and uniformly-distributed set of points within a polygon space.
    """
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_coords:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])

        if poly.contains(random_point):
            points.append(random_point)

    return points


def get_point_along_borderline(x, x_min_border, x_max_border, y_min_border, y_max_border, positive_quadrant):
    """
    Return the point along the sloped-border of the normal hand orientation
    """
    if positive_quadrant is False:
        slope = (y_max_border-y_min_border) / (0-x_min_border)
        b = -(slope*x_min_border-y_min_border)
    else:
        slope = (y_max_border-y_min_border) / -(x_max_border-0)
        b = -(slope*x_max_border-y_min_border)
    y = slope*x + b
    point = (x, y)

    return point


def generate_object_coordinates(num_coords,shape_name,orient_name,graspable_regions,difficulty,graspable_region_border):
    """
    Generate the set of initial object coordinates within the graspable range of the hand.

    """
    object_half_heights = {"CubeS": 0.04792,"CubeM": 0.05274,"CubeB": 0.05745,"CylinderM": 0.05498, "Vase1M": 0.05500}
    x_min, x_max, y_min, y_max = graspable_regions["x_min"], graspable_regions["x_max"], graspable_regions["y_min"], graspable_regions["y_max"]
    x_min_border, x_max_border, y_min_border, y_max_border = graspable_region_border["x_min"], graspable_region_border["x_max"], graspable_region_border["y_min"], graspable_region_border["y_max"]

    # Generate x,y coordinates within the graspable region
    if orient_name == "normal":
        if x_max <= 0:
            positive_quadrant = False
        else:
            positive_quadrant = True

        if difficulty == "hard":
            # Points with a hard difficulty are within the triangular edges of the grasping region
            if positive_quadrant is False:
                origin = get_point_along_borderline(x_max, x_min_border, x_max_border, y_min_border, y_max_border, positive_quadrant)
            else:
                origin = get_point_along_borderline(x_min, x_min_border, x_max_border, y_min_border, y_max_border, positive_quadrant)

            poly = Polygon([(x_min, y_min), origin, (x_max, y_min)])
        elif difficulty == "med" or difficulty == "easy":
            # Medium and easy points are defined within quadrilateral regions
            pt2 = get_point_along_borderline(x_min, x_min_border, x_max_border, y_min_border, y_max_border, positive_quadrant)
            pt3 = get_point_along_borderline(x_max, x_min_border, x_max_border, y_min_border, y_max_border, positive_quadrant)
            poly = Polygon([(x_min, y_min), pt2, pt3, (x_max, y_min)])
        else:
            poly = Polygon([(x_min, y_min), (0, y_max), (x_max, y_min)])
    else:
        poly = Polygon([(x_min, y_min), (x_min,y_max), (x_max, y_max), (x_max, y_min)])

    points = random_points_within(poly, num_coords)

    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]

    z = object_half_heights[shape_name]

    # Complete list of object coordinates
    obj_coords = [[x, y, z] for x, y in zip(x_coords, y_coords)]

    plot_points(obj_coords, orient_name, difficulty, graspable_region_border)

    return obj_coords


def plot_points(obj_coords, orient_name, difficulty, graspable_region_border):
    """
    Plot the frequency of each of the object-hand coordinates within a heatmap based on their difficulty
    """
    difficulty_colors = {"None": "black", "easy": "green", "med": "blue", "hard": "red"}
    x_min_border, x_max_border, y_min_border, y_max_border = graspable_region_border["x_min"], graspable_region_border["x_max"], graspable_region_border["y_min"], graspable_region_border["y_max"]

    if orient_name == "normal":
        border = Polygon([(x_min_border, y_min_border), (0, y_max_border), (x_max_border, y_min_border)])
    else:
        border = Polygon([(x_min_border, y_min_border), (x_min_border, y_max_border), (x_max_border, y_max_border), (x_max_border, y_min_border)])

    x_coords = [p[0] for p in obj_coords]
    y_coords = [p[1] for p in obj_coords]

    fig = plt.figure()
    fig.set_size_inches(11,8)   # Figure size
    fig.set_dpi(100)           # Pixel amount
    ax = fig.add_subplot(111)
    ax.set_xlim([-0.11, 0.11])
    ax.set_ylim([-0.11, 0.11])

    # Plot the boarder of the full graspable region
    plt.plot(*border.exterior.xy)

    # Plot points
    plt.scatter(x_coords, y_coords, s=1, c=difficulty_colors[difficulty])

    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MultipleLocator(0.01))   # Set axis tick locations
    ax.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))
    plt.show()


def generate_hand_rotations(num_coords,orient_name):
    """
    Generate a list of wrist coordinates for the robot hand
    """
    hand_rotations = np.zeros([num_coords, 3])
    rotation_variation = np.zeros([3])

    orient_euler_angles = {"normal": [-1.57, 0, -1.57], "rotated": [-0.7853, 0, -1.57], "top": [0, 0, -1.57]}

    for i in range(num_coords):
        # Added random variation +/- 5 degrees
        #rotation_variation[1:] = np.random.normal(-0.087, 0.087, 2) # Only add variation to the pitch/yaw
        rotation_variation = np.random.normal(-0.087, 0.087, 3)
        hand_rotations[i, :] = np.array(orient_euler_angles[orient_name]) + rotation_variation

    return hand_rotations


def write_transformations_to_file(obj_hand_poses, orient, shape, with_noise, difficulty):
    """
    Write the xml transformations for both the hand and the object.
    """
    if with_noise is False:
        noise_str = "no_noise"
    else:
        noise_str = "with_noise"

    if difficulty == "None":
        difficulty_str = ""
    else:
        difficulty_str = "_" + difficulty

    file_path = "./gym_kinova_gripper/envs/kinova_description/obj_hand_poses/" + noise_str + "/" + orient + "/" + shape + difficulty_str
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


def define_grasping_regions(all_orient_names):
    """
    Define the graspable regions and their estimated difficulty label
    """
    grasping_regions_by_difficulty = {
        "normal": {"easy": {}, "med": {}, "hard": {}},
        "rotated": {"easy": {}, "med": {}, "hard": {}},
        "top": {"easy": {}, "med": {}, "hard": {}}}
    percent_difficult = {"easy": .30, "med": .30, "hard": .40}
    graspable_regions = {"normal": {"x_min": -0.09, "x_max": 0.09, "y_min": -0.01, "y_max": 0.04},
                          "rotated": {"x_min":-0.09, "x_max": 0.09, "y_min": -0.06, "y_max": 0.02},
                         "top": {"x_min":-0.10, "x_max":0.10, "y_min":-0.04, "y_max":0.04}}  # x_min, x_max, y_min, y_max

    print("\nGRASPING REGION BOUNDARIES BY DIFFICULTY")

    for orient_name in all_orient_names:
        x_min, x_max = graspable_regions[orient_name]["x_min"], graspable_regions[orient_name]["x_max"]
        boundary_x_length = x_max - x_min
        hard_boundary_half_length = boundary_x_length * percent_difficult["hard"]/2
        med_boundary_half_length = boundary_x_length * percent_difficult["med"]/2
        easy_boundary_half_length = boundary_x_length * percent_difficult["easy"]/2

        # {"difficulty": [left_x_min, left_x_max, right_x_min, right_x_max], ...}
        grasping_regions_by_difficulty[orient_name]["easy"] = {"left_x_min": -easy_boundary_half_length, "left_x_max": 0, "right_x_min": 0, "right_x_max": easy_boundary_half_length}
        grasping_regions_by_difficulty[orient_name]["med"] = {"left_x_min": grasping_regions_by_difficulty[orient_name]["easy"]["left_x_min"] - med_boundary_half_length, "left_x_max": grasping_regions_by_difficulty[orient_name]["easy"]["left_x_min"],"right_x_min": grasping_regions_by_difficulty[orient_name]["easy"]["right_x_max"], "right_x_max": grasping_regions_by_difficulty[orient_name]["easy"]["right_x_max"] + med_boundary_half_length}
        grasping_regions_by_difficulty[orient_name]["hard"] = {"left_x_min": grasping_regions_by_difficulty[orient_name]["med"]["left_x_min"] - hard_boundary_half_length, "left_x_max": grasping_regions_by_difficulty[orient_name]["med"]["left_x_min"],"right_x_min": grasping_regions_by_difficulty[orient_name]["med"]["right_x_max"], "right_x_max": grasping_regions_by_difficulty[orient_name]["med"]["right_x_max"] + hard_boundary_half_length}

        print("For the ** {} ** hand orientation, the boundaries are:".format(orient_name))
        print("Overall [x_min,x_max]: [{},{}]; Boundary x length: {}".format(x_min,x_max,boundary_x_length))
        print("Easy: ",grasping_regions_by_difficulty[orient_name]["easy"].values())
        print("Med: ", grasping_regions_by_difficulty[orient_name]["med"].values())
        print("Hard: ", grasping_regions_by_difficulty[orient_name]["hard"].values(),"\n")

    return grasping_regions_by_difficulty, graspable_regions


if __name__ == "__main__":
    total_num_coords = 10000 # Number of HOV points to generate
    shape_names= ["CubeS","CubeM","CubeB", "CylinderM", "Vase1M"]
    orient_names= ["normal","rotated","top"]
    with_noise = True
    difficulty = "None"

    grasping_regions_by_difficulty, full_graspable_regions = define_grasping_regions(orient_names)

    for orient in orient_names:
        # Define the region based on the difficulty or produce points evenly throughout the whole region
        if difficulty != "None":
            difficulty_region = grasping_regions_by_difficulty[orient][difficulty]
            left_half = {"x_min": difficulty_region["left_x_min"], "x_max": difficulty_region["left_x_max"],
                         "y_min": full_graspable_regions[orient]["y_min"], "y_max": full_graspable_regions[orient]["y_max"]}
            right_half = {"x_min": difficulty_region["right_x_min"], "x_max": difficulty_region["right_x_max"],
                          "y_min": full_graspable_regions[orient]["y_min"], "y_max": full_graspable_regions[orient]["y_max"]}
            graspable_regions = [left_half, right_half]
        else:
            graspable_regions = [full_graspable_regions[orient]]

        # Split coordinates based on the number of regions we are producing coordinates in
        num_coords = int(total_num_coords/len(graspable_regions))

        for shape in shape_names:
            print("*** Generating {} coordinates with {} difficulty for the {} shape and {} hand orientation.".format(num_coords,difficulty,shape,orient))
            all_region_points = []
            for region in graspable_regions:
                obj_hand_poses = []
                # STEP 1: GENERATE OBJECT POSITIONS FOR EACH HAND ORIENTATION
                # Generate the object's initial coordinate positions (x,y,z)
                obj_coords = generate_object_coordinates(num_coords, shape_name=shape,orient_name=orient,graspable_regions=region,difficulty=difficulty,graspable_region_border=full_graspable_regions[orient])

                if with_noise is True:
                    # STEP 2: GENERATE HAND ROTATIONS IN RELATION TO EACH OBJECT
                    # Generate the robot hand (wrist) initial coordinate positions (x,y,z)
                    hand_rotations = generate_hand_rotations(num_coords,orient)

                    for obj, hand in zip(obj_coords, hand_rotations):
                        pose = np.append(obj, hand)
                        obj_hand_poses.append(pose)
                else:
                    obj_hand_poses = obj_coords

                # Include points from both the left and right regions
                all_region_points.extend(obj_hand_poses)

            plot_points(all_region_points, orient, difficulty, graspable_region_border=full_graspable_regions[orient])
            # STEP 3: WRITE EACH FULL SET OF OBJECT-HAND COORDINATES TO FILE
            write_transformations_to_file(all_region_points, orient, shape, with_noise, difficulty)

    # STEP 4: CHECK OBJECT-HAND COORDINATES FOR COLLISION -- use check_coordinates_for_collision.py
    # STEP 5: SPLIT OBJECT-HAND COORDINATES INTO TRAIN/EVAL/TEST SETS -- use separate_train_test_coords.py

    # Note: For testing purposes, call the following function with a specific object-hand pose
    # render_object_hand_pose(wrist_coords=[0,0,0],requested_shape=shape,hand_orientation=orient)