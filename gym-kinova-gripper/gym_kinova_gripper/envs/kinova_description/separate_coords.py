import glob
import os
import csv
import random

train_coords_dir = "./train_coords"
if not os.path.isdir(train_coords_dir):
  os.mkdir(train_coords_dir)

test_coords_dir = "./test_coords"
if not os.path.isdir(test_coords_dir):
  os.mkdir(test_coords_dir)

orientations = ["Top","Normal","Side"]

for hand_orientation in orientations:
	for coords_file in glob.glob('shape_coords/'+hand_orientation+'/*.txt'):
		with open(coords_file) as csvfile:
			data = [(float(x), float(y), float(z)) for x, y, z in csv.reader(csvfile, delimiter= ' ')]

		train_coords = []
		amount_training = len(data) * 0.9
		train_coords = [data.pop(random.randrange(len(data))) for _ in range(int(amount_training))]

		coords_file = os.path.basename(coords_file)
		train_coords_path = train_coords_dir+"/"+hand_orientation+"/"
		if not os.path.isdir(train_coords_path):
		  os.mkdir(train_coords_path)
		with open(train_coords_path+coords_file, 'w', newline='') as outfile:
			w_train = csv.writer(outfile)
			for coords in train_coords:
				w_train.writerow(coords)

		test_coords_path = test_coords_dir+"/"+hand_orientation+"/"
		if not os.path.isdir(test_coords_path):
		  os.mkdir(test_coords_path)
		with open(test_coords_path+coords_file, 'w', newline='') as outfile:
			w_test = csv.writer(outfile)
			for coords in data:
				w_test.writerow(coords)
