import glob
import os
import csv
import random
import argparse

""" Takes in a full dataset of coordinates (x,y,z) from txt file and splits into train and test sets """

def split_coords(all_data_filepath,with_noise):
    """ Split full coordinate dataset into train and test sets
    all_data_filepath: Filepath to full dataset to be split (Ex: all_shapes/)
    """
    # Create train and test output files
    train_coords_dir = all_data_filepath + "/train_coords"
    if not os.path.isdir(train_coords_dir):
        os.mkdir(train_coords_dir)

    eval_coords_dir = all_data_filepath + "/eval_coords"
    if not os.path.isdir(eval_coords_dir):
        os.mkdir(eval_coords_dir)

    test_coords_dir = all_data_filepath + "/test_coords"
    if not os.path.isdir(test_coords_dir):
        os.mkdir(test_coords_dir)

    orientations = ["normal","rotated","top"]

    for hand_orientation in orientations:
        shape_coords_path = all_data_filepath+"/shape_coords/"+hand_orientation
        print("File: ",shape_coords_path)
        for coords_file in glob.glob(shape_coords_path+'/*.txt'):
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
                        data.append([float(i[0]), float(i[1]), float(i[2])])

            coords_file = os.path.basename(coords_file)

            # Split data into train (80%), evaluation (10%) and test (10%) sets
            amount_training = len(data) * 0.8
            amount_evaluation = len(data) * 0.1
            amount_test = len(data) * 0.1

            # Pop the randomly-selected training coords from the list
            train_coords = []
            train_coords = [data.pop(random.randrange(len(data))) for _ in range(int(amount_training))]

            train_coords_path = train_coords_dir + "/" + hand_orientation + "/"
            if not os.path.isdir(train_coords_path):
                os.mkdir(train_coords_path)

            # Write train orientation noise to file
            with open(train_coords_path + coords_file, 'w', newline='') as outfile:
                w_train = csv.writer(outfile)
                for coords in train_coords:
                    w_train.writerow(coords)

            # Pop the randomly-selected evaluation coords from the list
            eval_coords = [data.pop(random.randrange(len(data))) for _ in range(int(amount_evaluation))]

            eval_coords_path = eval_coords_dir + "/" + hand_orientation + "/"
            if not os.path.isdir(eval_coords_path):
                os.mkdir(eval_coords_path)

            # Write train orientation noise to file (Everything left in data)
            with open(eval_coords_path + coords_file, 'w', newline='') as outfile:
                w_train = csv.writer(outfile)
                for coords in eval_coords:
                    w_train.writerow(coords)

            # Pop the randomly-selected test coords from the list
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
    args = parser.parse_args()

    if args.with_orientation_noise == "True":
        with_orientation_noise = True
    else:
        with_orientation_noise = False

    # Split coordinates from the full dataset into train and test sets
    split_coords(args.all_data_filepath, with_orientation_noise)