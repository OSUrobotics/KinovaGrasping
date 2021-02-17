import glob
import os
import csv
import random

""" Takes in orientation noise values (x,y,z) from txt file and splits into train and test sets """

# Create train and test output files
train_coords_dir = "./train_orientation_noise"
if not os.path.isdir(train_coords_dir):
  os.mkdir(train_coords_dir)

test_coords_dir = "./test_orientation_noise"
if not os.path.isdir(test_coords_dir):
  os.mkdir(test_coords_dir)

orientations = ["normal","side","top"]

for hand_orientation in orientations:
    for coords_file in glob.glob(hand_orientation+'/*.txt'):
        print("coords_file: ",coords_file)
        file_data = []
        with open(coords_file) as csvfile:
            file_data = [row for row in csv.reader(csvfile, delimiter=' ')]
            data = []
            for row in file_data:
                good_row = []
                for value in row:
                    if value != '':
                        good_row.append(float(value))
                data.append(good_row)

        # Split data into train (90%) and test (10%) sets
        train_coords = []
        amount_training = len(data) * 0.9
        train_coords = [data.pop(random.randrange(len(data))) for _ in range(int(amount_training))]

        coords_file = os.path.basename(coords_file)
        train_coords_path = train_coords_dir+"/"+hand_orientation+"/"
        if not os.path.isdir(train_coords_path):
            os.mkdir(train_coords_path)

        # Write train orientation noise to file
        with open(train_coords_path+coords_file, 'w', newline='') as outfile:
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
