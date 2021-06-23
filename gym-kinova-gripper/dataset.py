import numpy as np
import os
import random


class Dataset:
    def __init__(self, gui_selection):
        self.gui_data = gui_selection
        self.all_options_train = {}
        self.all_options_test = {}
        self.all_options = {}
        self.test_idx = None

    def _generate_all_options(self):
        """
        gui = {"Data set": {'Train': {"Orn": {"Normal": {"Shapes": {"Cube": {"Sizes": ['S', 'L']},
                                                                "Cylinder": {"Sizes": ['S']}}},
                                          "Top": {"Shapes": {"Cone1": {"Sizes": ['S']}}}},
                                  "Noise": True,
                                  "Data Points": 2500,
                                  "Controller": 'Policy'},
                        'Test': {"Orn": {"Normal": {"Shapes": {"Cube": {"Sizes": ['M']},
                                                               "Cylinder": {"Sizes": ['M', 'L']}}},
                                         "Top": {"Shapes": {"Cone1": {"Sizes": ['M', 'L']}}}},
                                 "Noise": True,
                                 "Data Points": 1500,
                                 "Controller": 'Policy'}}}
        """
        dataset_options = list(self.gui_data["Data set"].keys())
        orn_options = self.gui_data["Data set"][dataset_options[0]]["Orn"]
        i = 0

        self.all_options_train, self.test_idx = self._generate_options(orn_options, i, self.all_options_test)
        self.all_options.update({dataset_options[0]: {"Opns": self.all_options_train}})

        if len(dataset_options) > 1:
            orn_options = self.gui_data["Data set"][dataset_options[1]]["Orn"]
            # print("Orn:", orn_options)

            self.all_options_test, _ = self._generate_options(orn_options, self.test_idx, self.all_options_train)
            self.all_options.update({dataset_options[1]: {"Opns": self.all_options_test}})

    @staticmethod
    def _generate_options(options, idx, compare_with_dict):
        add_options = {}
        for orn in options.keys():
            for shape in options[orn]["Shapes"]:
                for size in options[orn]["Shapes"][shape]["Sizes"]:
                    if [orn, shape, size] not in compare_with_dict.values():
                        add_options.update({idx: [orn, shape, size]})
                        idx += 1
                    else:
                        for key, val in compare_with_dict.items():
                            if val == [orn, shape, size]:
                                add_options.update({key: [orn, shape, size]})
        return add_options, idx

    def _generate_balanced_latin_squares(self):
        for dataset in self.all_options.keys():
            key_list = list(self.all_options[dataset]["Opns"].keys())
            n = len(key_list)
            l = [[((j // 2 + 1 if j % 2 else n - j // 2) + i) % n + 1 for j in range(n)] for i in range(n)]
            if n % 2:  # Repeat reversed for odd n
                l += [seq[::-1] for seq in l]
            l_sq = np.asarray(l) - 1
            l_sq_correct = np.zeros(l_sq.shape)
            for i in range(l_sq.shape[0]):
                for j in range(l_sq.shape[1]):
                    l_sq_correct[i][j] = key_list[l_sq[i][j]]

            self.all_options[dataset].update({"L_Square": l_sq_correct})

    def _use_latin_square(self):
        for dataset in self.all_options.keys():
            num_data_points = self.gui_data["Data set"][dataset]["Data Points"]
            i_num_data_points = 0
            latin_square = self.all_options[dataset]["L_Square"]
            row_latin_square = latin_square.shape[0]
            col_latin_square = latin_square.shape[1]
            data_set_options = []
            while i_num_data_points < num_data_points:
                for row in range(0, row_latin_square):
                    col = 0
                    while col < col_latin_square and i_num_data_points < num_data_points:
                        # print("ROW {}, COL {} LEN {}".format(row, col, col_latin_square))
                        data_set_options.append(latin_square[row][col])
                        col += 1
                        i_num_data_points += 1

            self.all_options[dataset].update({"Bal_Data": np.asarray(data_set_options)})

    def _get_start_coords_files(self):
        orn = 0
        shape = 1
        size = 2
        for dataset in self.all_options.keys():
            if self.gui_data["Data set"][dataset]["Noise"]:
                is_there_noise = "/with_noise"
            else:
                is_there_noise = "/no_noise"
            idx = 0
            for key, value in list(self.all_options[dataset]["Opns"].items()):
                orn_folder = "/" + value[orn]
                shape_size_file = "/" + value[shape] + value[size] + ".txt"

                filename = "gym_kinova_gripper/envs/kinova_description/obj_hand_coords" + is_there_noise + "/shape_coords" + orn_folder + shape_size_file
                self.all_options[dataset]["Opns"][key].append(filename)
                idx += 1

    def _store_data_from_start_coord_files(self):
        start_iterator = 0
        iterator = start_iterator
        iterator_inc = 1
        for dataset in self.all_options.keys():
            for key, value in self.all_options[dataset]["Opns"].items():
                with open(value[3], 'r') as f:
                    array_coords = f.readlines()
                random.shuffle(array_coords)

                self.all_options[dataset]["Opns"][key].append(array_coords)
                self.all_options[dataset]["Opns"][key].append(start_iterator)
                self.all_options[dataset]["Opns"][key].append(iterator_inc)
                self.all_options[dataset]["Opns"][key].append(iterator)

    def _generate_data_set(self, train_fname, test_fname=None):
        compared = False
        iterator_start_shared = 1
        iterator_inc_shared = 2
        if os.path.exists(train_fname):
            os.remove(train_fname)
        if test_fname is not None:
            if os.path.exists(test_fname):
                os.remove(test_fname)

        if len(list(self.all_options.keys())) > 1:
            for dataset in self.all_options.keys():
                if not compared:
                    compared = True
                    if dataset == "Train":
                        compare_with_dataset = "Test"
                    elif dataset == "Test":
                        compare_with_dataset = "Train"
                    else:
                        print("Invalid dataset type")
                        raise KeyError
                if dataset != compare_with_dataset:
                    for key_idx in self.all_options[dataset]["Opns"].keys():
                        if key_idx in self.all_options[compare_with_dataset]["Opns"].keys():
                            self.all_options[dataset]["Opns"][key_idx][6] = iterator_inc_shared
                            self.all_options[compare_with_dataset]["Opns"][key_idx][6] = iterator_inc_shared
                            self.all_options[compare_with_dataset]["Opns"][key_idx][5] = iterator_start_shared
                            self.all_options[compare_with_dataset]["Opns"][key_idx][7] = iterator_start_shared

        for dataset in self.all_options.keys():
            num_data_points = 0
            if dataset == "Train":
                fname = train_fname
            elif dataset == "Test":
                fname = test_fname
            else:
                fname = None
            for key in self.all_options[dataset]["Bal_Data"]:
                # print("DATASET", dataset)
                # print("KEY", key)
                # print("PRINT", self.all_options[dataset]["Opns"][key])
                iterator = self.all_options[dataset]["Opns"][key][7]
                iterate_amt = self.all_options[dataset]["Opns"][key][6]
                total_array_len = len(self.all_options[dataset]["Opns"][key][4])
                if iterator >= total_array_len:
                    iterator = self.all_options[dataset]["Opns"][key][5]
                single_row_from_array = self.all_options[dataset]["Opns"][key][4][iterator]
                parsed_line = self._parse_line(single_row_from_array, dataset, num_data_points, key)
                if fname is not None:
                    self._write_to_file(parsed_line, fname)
                self.all_options[dataset]["Opns"][key][7] = iterator + iterate_amt
                num_data_points += 1

    def _parse_line(self, single_line, dataset, data_point, key):
        split_up_line = single_line.strip().split()

        idx = data_point
        # print("DATASET:", dataset, key)
        noise = self.gui_data["Data set"][dataset]["Noise"]
        orn = self.all_options[dataset]["Opns"][key][0]
        orn_data_1 = split_up_line[3]
        orn_data_2 = split_up_line[4]
        orn_data_3 = split_up_line[5]
        shape_size = self.all_options[dataset]["Opns"][key][1] + self.all_options[dataset]["Opns"][key][2]
        start_pos_1 = split_up_line[0]
        start_pos_2 = split_up_line[1]
        start_pos_3 = split_up_line[2]
        controller = self.gui_data["Data set"][dataset]["Controller"]

        return idx, noise, orn, orn_data_1, orn_data_2, orn_data_3, shape_size, start_pos_1, start_pos_2, start_pos_3, controller

    def _write_to_file(self, data, filename):
        data_string = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(data[0], data[1], data[2], data[3], data[4],
                                                                            data[5], data[6], data[7], data[8], data[9],
                                                                            data[10])

        with open(filename, 'a') as f:
            f.write(data_string)

    def call_of_duty(self, train_file, test_file=None):
        self._generate_all_options()
        self._generate_balanced_latin_squares()
        self._use_latin_square()
        self._get_start_coords_files()
        self._store_data_from_start_coord_files()
        self._generate_data_set(train_file, test_file)


if __name__ == '__main__':
    # TODO: Rearrange incoming  data to go from dataset type to rest json format - DONE
    # TODO: Check for same shape/size/orn in test and  train. Give them the same idx label - DONE
    # TODO: Check for no repititions in test and train - DONE
    # TODO: Branch out for noisy poses- DONE

    """
    gui = {"Data set": {'Train': {"Orn": {"Normal": {"Shapes": {"Cube": {"Sizes": ['S', 'B']},
                                                                "Cylinder": {"Sizes": ['S']}}},
                                          "Top": {"Shapes": {"Cone1": {"Sizes": ['S']}}}},
                                  "Noise": True,
                                  "Data Points": 20000,
                                  "Controller": 'Policy'},
                        'Test': {"Orn": {"Normal": {"Shapes": {"Cube": {"Sizes": ['M']},
                                                               "Cylinder": {"Sizes": ['M', 'B']}}},
                                         "Top": {"Shapes": {"Cone1": {"Sizes": ['M', 'B']}}}},
                                 "Noise": True,
                                 "Data Points": 5000,
                                 "Controller": 'Policy'}}}
    """

    """
    gui = {"Data set": {'Train': {"Orn": {"Normal": {"Shapes": {"Cube": {"Sizes": ['M']}}}},
                                  "Noise": False,
                                  "Data Points": 10000,
                                  "Controller": 'policy'},
                        'Test': {"Orn": {"Normal": {"Shapes": {"Cube": {"Sizes": ['M']}}}},
                                 "Noise": False,
                                 "Data Points": 5000,
                                 "Controller": 'policy'}}}
    """

    gui = {"Data set": {'Train': {"Orn": {"Normal": {"Shapes": {"Cube": {"Sizes": ['M']}}}},
                                  "Noise": True,
                                  "Data Points": 5000,
                                  "Controller": 'position-dependent'},
                        'Test': {"Orn": {"Normal": {"Shapes": {"Cube": {"Sizes": ['M']}}}},
                                 "Noise": True,
                                 "Data Points": 5000,
                                 "Controller": 'position-dependent'}}}

    exp1 = Dataset(gui)
    train_file = 'new_train.txt'
    test_file = 'new_test.txt'
    exp1.call_of_duty(train_file, test_file)

