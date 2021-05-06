import numpy as np
import os


class RLExpSetup:
    xml_file = {"CubeS": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1.xml",
                "CubeM": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mbox.xml",
                "CubeB": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_bbox.xml",
                "Cube45S": None,
                "Cube45M": None,
                "Cube45B": None,
                "CylinderS": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_scyl.xml",
                "CylinderM": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mcyl.xml",
                "CylinderB": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_bcyl.xml",
                "HourS": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_shg.xml",
                "HourM": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mhg.xml",
                "HourB": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_bhg.xml",
                "Cone1S": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_scone1.xml",
                "Cone1M": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mcone1.xml",
                "Cone1B": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_bcone1.xml",
                "Cone2S": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_scone2.xml",
                "Cone2M": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mcone2.xml",
                "Cone2B": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_bcone2.xml",
                "VaseS": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_svase.xml",
                "VaseM": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mvase.xml",
                "VaseB": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_bvase.xml",
                "Vase2S": None,
                "Vase2M": None,
                "Vase2B": None,
                "BottleS": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_sbottle.xml",
                "BottleM": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mbottle.xml",
                "BottleB": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_bbottle.xml",
                "TBottleS": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_stbottle.xml",
                "TBottleM": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mtbottle.xml",
                "TBottleB": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_btbottle.xml",
                "RBowlS": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_sRectBowl.xml",
                "RBowlM": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mRectBowl.xml",
                "RBowlB": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_bRectBowl.xml",
                "BowlS": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_sRoundBowl.xml",
                "BowlM": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mRoundBowl.xml",
                "BowlB": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_bRoundBowl.xml",
                "LemonS": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_slemon.xml",
                "LemonM": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_mlemon.xml",
                "LemonB": "gym-kinova-gripper/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_blemon.xml"}

    def __init__(self, file_name):
        """
        Expects data-set in to be a file with each row of the following format:
        obj_name, obj_x, obj_y, obj_z, hand_orn_type, hand_orn_x, hand_orn_y, hand_orn_z

        Expects the gui_params to be a dict of the following format:
        {"Noise": True/False,}
        """
        self.file_name = file_name
        with open(self.file_name, 'r') as self.file_pointer:
            self.dataset = self.file_pointer.readlines()
        self._iterator = 0
        self.metadata = None
        self._len = len(self.dataset)

    def get_data_next_episode(self):
        if self._iterator < self._len:
            extract_data = self.dataset[self._iterator]
            self.metadata = self.generate_metadata(extract_data)
            self._iterator += 1
        else:
            print("COMPLETE")
            self.metadata = None
        return self.metadata

    def get_data_requested_episode(self, ep_num):
        """
        Extracts data from gui  parameters and dataset to  create meta data  for  current episode
        :param ep_num: The episode number to generate data for
        :return: metadata_dict: Dictionary of all data
        """
        if ep_num < self._len:
            extract_data = self.dataset[ep_num]
            self.metadata = self.generate_metadata(extract_data)
        else:
            print("Requested Episode Number exceeds data set length")
            self.metadata = None
        return self.metadata

    def generate_metadata(self, data_line):
        data_line = ''.join(data_line.split()).split(',')
        print("DATA:", data_line)
        metadata = {"idx": int(data_line[0]), "Noise": bool(data_line[1]), "Orn": data_line[2],
                    "Orn_Values": [float(data_line[3]),
                                   float(data_line[4]),
                                   float(data_line[5])],
                    "Shape": data_line[6],
                    "Start_Values": [float(data_line[7]), float(data_line[8]), float(data_line[9])], "xml_file":
                        RLExpSetup.xml_file[data_line[6]], "Controller": data_line[10]}
        return metadata




if __name__ == '__main__':

    # f_name = "/Users/asar/Desktop/Grimm's Lab/Grasping/Codes/KinovaGrasping/gym-kinova-gripper/train_trial.txt"
    f_name_train = "gym-kinova-gripper/cor_train_trial_new.txt"
    f_name_test = "gym-kinova-gripper/cor_test_trial_new.txt"

    train = RLExpSetup(f_name_train)
    test = RLExpSetup(f_name_test)
    # meta_data = exp1.get_data_requested_episode(3)
    # print("Meta data 1: {}".format(meta_data))

    for i in range (0, 15):
        meta_data = train.get_data_next_episode()
        print("Meta data Train{}".format(meta_data))
        meta_data = test.get_data_next_episode()
        print("Meta data Test{}".format(meta_data))
