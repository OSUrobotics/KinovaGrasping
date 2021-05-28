import glob
import os
import csv
import random
import argparse

""" Takes in a full dataset of coordinates (x,y,z) from txt file and splits into train and test sets """

def split_coords(all_data_filepath,with_noise,split_type):
    """ Split full coordinate dataset into train and test sets
    all_data_filepath: Filepath to full dataset to be split (Ex: all_shapes/)
    """
    # Create train and test output files
    train_coords_dir = "./train_coords"
    if not os.path.isdir(train_coords_dir):
        os.mkdir(train_coords_dir)

    eval_coords_dir = "./eval_coords"
    if not os.path.isdir(eval_coords_dir):
        os.mkdir(eval_coords_dir)

    test_coords_dir = "./test_coords"
    if not os.path.isdir(test_coords_dir):
        os.mkdir(test_coords_dir)

    orientations = ["normal","rotated","top"]

    for hand_orientation in orientations:
        for coords_file in glob.glob(all_data_filepath+hand_orientation+'/*.txt'):
            data = []
            with open(coords_file) as csvfile:
                checker = csvfile.readline()
                if ',' in checker:
                    delim = ','
                else:
                    delim = ' '
                reader = csv.reader(csvfile, delimiter=delim)
                for i in reader:
                    if with_noise is True:
                        # Object x, y, z coordinates, followed by corresponding hand orientation x, y, z coords
                        data.append([float(i[0]), float(i[1]), float(i[2]), float(i[3]), float(i[4]), float(i[5])])
                    else:
                        # Hand orientation is set to (0, 0, 0) if no orientation is selected
                        data.append([float(i[0]), float(i[1]), float(i[2]), 0, 0, 0])

            coords_file = os.path.basename(coords_file)

            if split_type == "eval":
                eval_coords = [data.pop(random.randrange(len(data))) for _ in range(500)]

                eval_coords_path = eval_coords_dir + "/" + hand_orientation + "/"
                if not os.path.isdir(eval_coords_path):
                    os.mkdir(eval_coords_path)

                # Write train orientation noise to file
                with open(eval_coords_path + coords_file, 'w', newline='') as outfile:
                    w_train = csv.writer(outfile)
                    for coords in eval_coords:
                        w_train.writerow(coords)

            elif split_type == "train_test":
                # Split data into train (90%) and test (10%) sets
                train_coords = []
                amount_training = len(data) * 0.9
                train_coords = [data.pop(random.randrange(len(data))) for _ in range(int(amount_training))]

                train_coords_path = train_coords_dir + "/" + hand_orientation + "/"
                if not os.path.isdir(train_coords_path):
                    os.mkdir(train_coords_path)

                # Write train orientation noise to file
                with open(train_coords_path + coords_file, 'w', newline='') as outfile:
                    w_train = csv.writer(outfile)
                    for coords in train_coords:
                        w_train.writerow(coords)

                test_coords_path = test_coords_dir+"/"+hand_orientation+"/"
                if not os.path.isdir(test_coords_path):
                    os.mkdir(test_coords_path)

                # Write test orientation noise to file
                with open(test_coords_path+coords_file, 'w', newline='') as outfile:
                    w_test = csv.writer(outfile)
                    for coords in data:
                        w_test.writerow(coords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_data_filepath", type=str, action='store', default="./")
    parser.add_argument("--with_orientation_noise", type=str, action='store', default="True")
    parser.add_argument("--split_type", type=str, action='store', default="train_test")
    args = parser.parse_args()

    if args.with_orientation_noise == "True":
        with_orientation_noise = True
    else:
        with_orientation_noise = False

    # Split coordinates from the full dataset into train and test sets
    split_coords(args.all_data_filepath, with_orientation_noise, args.split_type)