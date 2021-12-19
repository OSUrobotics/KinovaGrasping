import os, sys
import csv
import random
import argparse
import random
from pathlib import Path
from label_obj_hand_coords_by_difficulty import read_obj_hand_pose_dict_list, write_obj_hand_pose_dict_list


""" Takes in a full dataset of object-hand coordinates from a text file or labelled csv file 
and splits into train, evaluation, and test sets """


def read_coords_from_text_file(with_noise,coords_filepath,filename):
    """
    Separate coordinates from a text file into train, eval, and test sets.
    """
    data = []
    with open(coords_filepath + filename) as csvfile:
        checker = csvfile.readline()
        if ',' in checker:
            delim = ','
        else:
            delim = ' '
    # Go back to the top of the file after checking for the delimiter
    with open(coords_filepath + filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=delim)
        for i in reader:
            if with_noise is True:
                # Object x, y, z coordinates, followed by corresponding hand orientation x, y, z coords
                data.append([float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4]), float(i[5])])
            else:
                # Hand orientation is set to (0, 0, 0) if no orientation is selected
                data.append([float(i[0]), float(i[1]), float(i[2])])

    return data


def separate_coords_randomly(amount_training,amount_evaluation,amount_test,data):
    """
    Randomly separate coordinates per coordinate dataset type, return train/eval/test datasets
    """
    # Pop the randomly-selected training coords from the list
    train_coords = [data.pop(random.randrange(len(data))) for _ in range(int(amount_training))]

    # Pop the randomly-selected evaluation coords from the list
    eval_coords = [data.pop(random.randrange(len(data))) for _ in range(int(amount_evaluation))]

    # After the training and eval coords have been popped from the list, only test coords are left
    test_coords = data

    return train_coords, eval_coords, test_coords


def separate_coords_by_difficulty(amount_training,amount_evaluation,amount_test,data):
    """
    Separate coordinates into train/eval/test sets based on the difficulty
    """
    all_easy_coords = [obj_coord for obj_coord in data if obj_coord["difficulty"] == "easy"]
    all_med_coords = [obj_coord for obj_coord in data if obj_coord["difficulty"] == "med"]
    all_hard_coords = [obj_coord for obj_coord in data if obj_coord["difficulty"] == "hard"]

    # Randomly shuffle coords so they are distributed among each difficulty 'region'
    random.shuffle(all_easy_coords)
    random.shuffle(all_med_coords)
    random.shuffle(all_hard_coords)

    dataset_amounts = {"train":{"easy":int(amount_training*.40),"med":int(amount_training*.40),"hard":int(amount_training*.20)}, "eval":{"easy":40,"med":60,"hard":40}, "test":{"easy":int(amount_test*.30),"med":int(amount_test*.30),"hard":int(amount_test*.40)}}

    # Training: 40% easy, 40% med, 20% hard
    train_easy = [all_easy_coords.pop() for _ in range(dataset_amounts["train"]["easy"])]
    train_med = [all_med_coords.pop() for _ in range(dataset_amounts["train"]["med"])]
    train_hard = [all_hard_coords.pop() for _ in range(dataset_amounts["train"]["hard"])]
    train_coords = train_easy + train_med + train_hard

    # Evaluation:  40 easy /60 med /40 hard
    eval_coords = [all_easy_coords.pop() for _ in range(dataset_amounts["eval"]["easy"])] + [all_med_coords.pop() for _ in range(dataset_amounts["eval"]["med"])] + [all_hard_coords.pop() for _ in range(dataset_amounts["eval"]["hard"])]

    # Test: what's left?
    test_coords = [all_easy_coords.pop() for _ in range(dataset_amounts["test"]["easy"])] + [all_med_coords.pop() for _ in range(dataset_amounts["test"]["med"])] + [all_hard_coords.pop() for _ in range(dataset_amounts["test"]["hard"])]

    return train_coords, eval_coords, test_coords


def create_output_paths(all_data_filepath,hand_orientation,shape_name):
    """
    Create train and test output files
    """
    all_dataset_dirs = ["/train_coords/","/eval_coords/","/test_coords/"]

    for new_dir in all_dataset_dirs:
        if new_dir is not None:
            new_path = Path(all_data_filepath + new_dir + "/" + hand_orientation + "/" + shape_name + "/labelled_coords/")
            new_path.mkdir(parents=True, exist_ok=True)


def write_coords(all_data_filepath, hand_orientation, shape_name, train_coords, eval_coords, test_coords, use_labelled_data):
    """
    Write object-hand coordinates to a text file
    """
    all_dataset_dirs = ["/train_coords/", "/eval_coords/", "/test_coords/", "/shape_coords/"]
    all_coords = train_coords + eval_coords + test_coords
    coord_sets = [train_coords, eval_coords, test_coords, all_coords]

    for coords,dataset_dir in zip(coord_sets,all_dataset_dirs):
        if use_labelled_data is True:
            labelled_data_dir = all_data_filepath + dataset_dir + hand_orientation + "/" + shape_name + "/labelled_coords/"
            write_obj_hand_pose_dict_list(labelled_data_dir, filename="labelled_obj_hand_coords.csv", labelled_obj_hand_coords=coords)
        else:
            print("Write to text file")
            """
            with open(all_data_filepath + dataset_dir + hand_orientation + "/" + shape_name + ".txt", 'w', newline='') as outfile:
                coord_writer = csv.writer(outfile)
                for c in coords:
                    coord_writer.writerow(c)
            """


def split_coords(all_data_filepath,all_shapes,all_orientations,with_noise,use_labelled_data,separate_by_difficulty):
    """ Split full coordinate dataset into train and test sets
    all_data_filepath: Filepath to full dataset to be split (Ex: all_shapes/)
    """

    for hand_orientation in all_orientations:
        for shape_name in all_shapes:
            # Create new directories per shape/hand orientation as needed
            create_output_paths(all_data_filepath, hand_orientation, shape_name)

            # Read in coordinate data
            coords_filepath = all_data_filepath+"/shape_coords/"+hand_orientation+"/"
            print("File: ",coords_filepath)

            # Read in object-hand pose coordinates
            if use_labelled_data is True:
                coords_filepath += shape_name + "/labelled_coords/"
                data, _ = read_obj_hand_pose_dict_list(coords_filepath, num_coords=None, filename="labelled_obj_hand_coords.csv")
            else:
                data = read_coords_from_text_file(with_noise, coords_filepath, filename=shape_name+".txt")

            # Split data into train (80%), evaluation (3.5%) and test (16.5%) sets
            total_num_coords = 1600 #4000
            amount_training = 1000
            amount_evaluation = 140
            amount_test = 460

            if separate_by_difficulty is True:
                train_coords, eval_coords, test_coords = separate_coords_by_difficulty(amount_training,amount_evaluation,amount_test,data)
            else:
                train_coords, eval_coords, test_coords = separate_coords_randomly(amount_training,amount_evaluation,amount_test,data)

            # Write object-hand pose coordinates to file
            write_coords(all_data_filepath, hand_orientation, shape_name, train_coords, eval_coords, test_coords, use_labelled_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_data_filepath", type=str, action='store', default="./gym_kinova_gripper/envs/kinova_description/obj_hand_coords/")
    parser.add_argument("--with_orientation_noise", type=str, action='store', default="True")
    args = parser.parse_args()

    all_shapes = ["CubeM"]
    all_orientations = ["normal"] #, "rotated", "top"]
    use_labelled_data = True
    separate_by_difficulty = True

    if args.with_orientation_noise == "True":
        with_orientation_noise = True
        noise_str = "/with_noise/"
    else:
        with_orientation_noise = False
        noise_str = "/no_noise/"

    all_data_filepath = args.all_data_filepath + noise_str

    # Split coordinates from the full dataset into train and test sets
    split_coords(all_data_filepath,all_shapes,all_orientations,with_orientation_noise,use_labelled_data,separate_by_difficulty)