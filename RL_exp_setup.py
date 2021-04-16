class RLExpSetup:

    def __init__(self, file_name, gui_params):
        """
        Expects data-set in to be a file with each row of the following format:
        obj_name, obj_x, obj_y, obj_z, hand_orn_type, hand_orn_x, hand_orn_y, hand_orn_z

        Expects the gui_params to be a dict of the following format:
        {"Noise": True/False,}
        """
        self.file_name = file_name
        self.gui_params = gui_params
        with open(self.file_name, 'r') as self.file_pointer:
            self.dataset = self.file_pointer.readlines()

    def generate_metadata_for_episode(self, ep_num):
        """
        Extracts data from gui  parameters and dataset to  create meta data  for  current episode
        :param ep_num: The episode number to generate data for
        :return: metadata_dict: Dictionary of all data
        """
        curr_data = self.dataset[ep_num]
        curr_data = curr_data.rstrip("\n")
        curr_data = curr_data.split(',')
        print("GUI", self.gui_params)
        print("EP DATA", curr_data)
        metadata_dict = {"Episode": ep_num, "Obj Shape": curr_data[0][:-1], "Obj Size": curr_data[0][-1],
                         "Obj Starting Pose": [float(curr_data[1]), float(curr_data[2]), float(curr_data[3])],
                         "Hand Orn": curr_data[4], "Hand Noisy Data": self.gui_params["Noise"], "Hand Starting Pose":
                             [float(curr_data[5]), float(curr_data[6]), float(curr_data[7])]}
        return metadata_dict


if __name__ == '__main__':
    f_name = "train_trial.txt"
    gui_param = {"Noise": True}
    exp1 = RLExpSetup(f_name, gui_param)
    meta_data = exp1.generate_metadata_for_episode(3)
    print("Meta data: {}".format(meta_data))
