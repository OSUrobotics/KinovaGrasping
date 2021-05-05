import numpy as np
import os

class Dataset:
    def __init__(self, gui_selection):
        self.gui_data = gui_selection

    def _generate_all_options(self):
        """
        {"Orn": ['Top', 'Normal'],
        "Shapes": [['Cube', 'Cylinder'],['Cube', 'Cylinder']],
        "Sizes": [['S', 'L'],['S', 'L']],
        "Noise": True
        "Data Points": [5000, 3000]
        "Data set": ['Train', 'Test'],
        "Controller": 'Policy'}
        """
        all_options = {}
        orn_options = self.gui_data["Orn"]
        i = 0
        for orn in range(0, len(orn_options)):
            for shape in range(0, len(self.gui_data["Shapes"][orn])):
                for size in range(0, len(self.gui_data["Sizes"][orn][shape])):
                    all_options.update(
                        {i: [self.gui_data["Orn"][orn], self.gui_data["Shapes"][orn][shape],
                             self.gui_data["Sizes"][orn][shape][size]]})
                    i += 1
        return all_options

    def _use_latin_square(self, latin_square, type_of_data):
        num_data_points = type_of_data
        i_num_data_points = 0
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

        return np.asarray(data_set_options)

    def _generate_balanced_latin_squares(self, all_possible_dict):
        n = len(list(all_possible_dict.keys()))
        l = [[((j // 2 + 1 if j % 2 else n - j // 2) + i) % n + 1 for j in range(n)] for i in range(n)]
        if n % 2:  # Repeat reversed for odd n
            l += [seq[::-1] for seq in l]
        return np.asarray(l) - 1

    def _get_start_coords_files(self, all_options):
        orn = 0
        shape = 1
        size = 2
        file_dict = {}
        if self.gui_data["Noise"]:
            is_there_noise = "/with_noise"
        else:
            is_there_noise = "/no_noise"
        idx = 0
        for value in list(all_options.values()):
            orn_folder = "/" + value[orn]
            shape_size_file = "/" + value[shape] + value[size] + ".txt"

            filename = "gym_kinova_gripper/envs/kinova_description/obj_hand_coords" + is_there_noise + "/shape_coords" + orn_folder + shape_size_file
            file_dict.update({idx: filename})
            idx += 1
        return file_dict

    def _store_data_from_start_coord_files(self, start_coord_file_dict):
        start_coords_dict = {}
        for key in start_coord_file_dict.keys():
            with open(start_coord_file_dict[key], 'r') as f:
                array_coords = f.readlines()
            start_coords_dict.update({key: array_coords})
        return start_coords_dict

    def _generate_data_set(self, all_options, start_values_dict, latin_square_train, train_fname,
                          latin_square_test=None, test_fname=None):
        done_train = False
        done_test = False
        i_train = 0
        i_test = 0
        if latin_square_test is None:
            done_test = True
            idx_train = 0
            idx_test = 0

        else:
            if self.gui_data["Data set"][0] == 'Train':
                idx_train = 0
                idx_test = 1
            else:
                idx_train = 1
                idx_test = 0
        train_data_set = []
        test_data_set = []
        idx_for_train = 0
        idx_for_test = 0
        all_idx_for_train_list = []
        all_idx_for_test_list = []

        if os.path.exists(train_fname):
            os.remove(train_fname)

        if test_fname is not None:
            if os.path.exists(test_fname):
                os.remove(test_fname)

        while not done_train or not done_test:
            if i_train < self.gui_data["Data Points"][idx_train]:
                idx_for_train, all_idx_for_train_list, data_line_train = self._get_new_data(i_train, idx_for_train,
                                                                                           all_idx_for_train_list,
                                                                                           start_values_dict,
                                                                                           latin_square_train,
                                                                                           all_options)
                train_data_set.append(data_line_train)
                i_train += 1
                self._write_to_file(data_line_train, train_fname)

            else:
                done_train = True

            if i_test < self.gui_data["Data Points"][idx_test]:
                if latin_square_test is not None:
                    idx_for_test, all_idx_for_test_list, data_line_test = self._get_new_data(i_test, idx_for_test,
                                                                                            all_idx_for_test_list,
                                                                                            start_values_dict,
                                                                                            latin_square_test,
                                                                                            all_options)
                    test_data_set.append(data_line_test)
                    i_test += 1
                    self._write_to_file(data_line_test, test_fname)
            else:
                done_test = True

        return train_data_set, test_data_set

    def _get_new_data(self, required_example, idx, idx_list, start_vals_dict, latin_sq_data, options_dict):
        a = 1
        if idx == len(start_vals_dict[latin_sq_data[required_example]]):
            idx = idx_list[0]
            a = None
            idx_list.append(True)
        if len(idx_list) != 0:
            if idx_list[-1] and a is not None:
                if idx >= (len(idx_list) - 1):
                    recycled_idx = idx_list[0]
                else:
                    recycled_idx = idx_list[idx]
                one_line = start_vals_dict[latin_sq_data[required_example]][recycled_idx]
            else:
                one_line = start_vals_dict[latin_sq_data[required_example]][idx]
            if not idx_list[-1]:
                idx_list.append(idx)
        else:
            one_line = start_vals_dict[latin_sq_data[required_example]][idx]
            idx_list.append(idx)
        option = options_dict[latin_sq_data[required_example]]
        noise = self.gui_data["Noise"]
        controller = self.gui_data["Controller"]
        parsed_line = self._parse_line(one_line, noise, option, required_example, controller)

        return idx + 1, idx_list, parsed_line

    def _parse_line(self, single_line, is_there_noise, selected_option, index, which_controller):
        split_up_line = single_line.strip().split()
        for obs in range(0, len(split_up_line)):
            split_up_line[obs] = float(split_up_line[obs])
        return index, is_there_noise, selected_option[0], split_up_line[3], split_up_line[4], split_up_line[5], \
               selected_option[1] + selected_option[2], split_up_line[0], split_up_line[1], split_up_line[2], \
               which_controller

    def _write_to_file(self, data, filename):
        data_string = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(data[0], data[1], data[2], data[3], data[4],
                                                                            data[5], data[6], data[7], data[8], data[9],
                                                                            data[10])

        with open(filename, 'a') as f:
            f.write(data_string)



if __name__ == '__main__':
    # TODO: Rearrange incoming  data to go from dataset type to rest json format
    # TODO: Check for same shape/size/orn in test and  train. Give them the same idx label
    # TODO: Check for no repititions in test and train
    # TODO: Branch out for noisy poses

    gui = {"Orn": ['Top', 'Normal'], "Shapes": [['Cube', 'Cylinder', 'Cone1'], ['Cube45']], "Sizes":
        [[['S', 'B'], ['S', 'B'], ['S', 'M', 'B']],[['S', 'B'], ['S', 'B']]], "Noise": True, "Data Points": [20000, 5000], "Data set": ['Train', 'Test'],
           "Controller": 'Policy'}
    # gui = {"Orn": ['Normal'], "Shapes": [['Cube', 'Cylinder']], "Sizes":
    #     [[['S', 'L'], ['S']]], "Noise": True, "Data Points": 2500, "Data set": ['Train', 'Test'],
    #        "Controller": 'Policy'}

    exp1 = Dataset(gui)

    all_options = exp1._generate_all_options()
    print(all_options)
    latin_square_1 = exp1._generate_balanced_latin_squares(all_options)
    print(latin_square_1)
    ans2_train = exp1._use_latin_square(latin_square_1, gui["Data Points"][0])
    print(ans2_train, len(ans2_train))
    ans2_test = exp1._use_latin_square(latin_square_1, gui["Data Points"][1])
    print(ans2_test, len(ans2_test))
    start_files = exp1._get_start_coords_files(all_options)
    print(start_files)
    start_dict = exp1._store_data_from_start_coord_files(start_files)
    print(start_dict)
    # # print(ans4[8])
    # # for i in range(0, len(ans4)):
    # #     print(len(ans4[i]))
    train_file = 'train_trial_new.txt'
    test_file = 'test_trial_new.txt'
    ans5_train, ans5_test = exp1._generate_data_set(all_options, start_dict, ans2_train, train_file, ans2_test, test_file)
    print(ans5_train)