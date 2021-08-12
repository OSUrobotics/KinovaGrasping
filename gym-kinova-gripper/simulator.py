from mujoco_py import MjViewer, load_model_from_path, MjSim
import os
import numpy as np

class Simulator():
    def __init__(self):
        self.file_dir = os.path.dirname(os.path.realpath(__file__))
        self.filename = "/gym_kinova_gripper/envs/kinova_description/j2s7s300_end_effector_v1_CubeS.xml" #%TODO Simulator class
        self._model = load_model_from_path(self.file_dir + self.filename)

        self._sim = MjSim(self._model)
        self._viewer = None
        #self._sim.data.ctrl = 0 # Determine how to handle the data structure

        # This data should be sent by the environment class
        # [xloc, yloc, zloc, f1prox, f2prox, f3prox, obj_x, obj_y, obj_z]
        self.sim_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    def get_sim(self):
        """ Returns the simulation (Mujoco MjSim). MjSim represents a running simulation including its state. """
        return self._sim

    def load_sim(self):
        """ Loads the simulator based on the currently set model """
        self._model = load_model_from_path(self.file_dir + self.filename)
        self._sim = MjSim(self._model)

    def set_filepath(self,file_dir,filename):
        """ Set the file directory and file path for the simulation xml file"""
        self.file_dir = file_dir
        self.filename = filename

    def get_viewer(self):
        """ Returns the simulation (Mujoco MjSim). MjSim represents a running simulation including its state. """
        return self._viewer

    def _set_state(self, states):
        """ Sets the joint positions (qpos) and performs a forward step in the simulator (Mujoco MjSim).
        states: array of coordinate values for the hand, finger, and object positions
        """
        # Hand (wrist) center coordinates
        self._sim.data.qpos[0] = states[0] # xloc
        self._sim.data.qpos[1] = states[1] # yloc
        self._sim.data.qpos[2] = states[2] # zloc

        # Finger proximal joint coordinates
        self._sim.data.qpos[3] = states[3] # f1prox
        self._sim.data.qpos[5] = states[4] # f2prox
        self._sim.data.qpos[7] = states[5] # f3prox

        # Object (center) coordinates: obj_x, obj_y, obj_z,
        self._sim.data.set_joint_qpos("object", [states[6], states[7], states[8], 1.0, 0.0, 0.0, 0.0])

        # Forward dynamics: advances the simulation but does not integrate in time
        self._sim.forward()

    def render(self):
        """ Renders the (Mujoco) simulation in a new window. The simulation viewer is set in a paused state unless set to true. """
        setPause=False
        if self._viewer is None:
            self._viewer = MjViewer(self._sim)
            self._viewer._paused = setPause
        self._viewer.render()
        if setPause:
            self._viewer._paused=True

    def set_viewer(self):
        # Initialize the simulation viewer (Mujoco) if not already
        if self._viewer is None:
            self._viewer = MjViewer(self._sim)

    def add_site(self,world_site_coords,keep_sites=False):
        """ Adds bright lines in the simulator for viewing coordinate locations by adding sites to the xml file
        world_site_coords: Coordinate location within the world where lines will be placed within the simulation (Mujoco)
        keep_sites: Set to true to add sites (lines) to the xml file
        """
        if not(keep_sites):
            self.site_count=0
        xml_file=open(self.file_dir+self.filename,"r")
        xml_contents=xml_file.read()
        xml_file.close()
        a=xml_contents.find('<site name="site{self.site_count}" type="cylinder" size="0.001 0.2" rgba="25 0.5 0.7 1" pos="{world_site_coords[0]} {world_site_coords[1]} {world_site_coords[2]}" euler="0 1.5707963267948966 0"/>\n')
        if a!=-1:
            starting_point=xml_contents.find('<body name="root" pos="0 0 0">')
            site_point=xml_contents.find('\n',starting_point)
            site_text=f'            <site name="site{self.site_count}" type="cylinder" size="0.001 0.2" rgba="25 0.5 0.7 1" pos="{world_site_coords[0]} {world_site_coords[1]} {world_site_coords[2]}" euler="0 0 0"/>\n'
            self.site_count+=1
            second_site_text=f'            <site name="site{self.site_count}" type="cylinder" size="0.001 0.2" rgba="25 0.5 0.7 1" pos="{world_site_coords[0]} {world_site_coords[1]} {world_site_coords[2]}" euler="0 1.5707963267948966 0"/>\n'
            self.site_count+=1
            new_thing=xml_contents[0:site_point+1]+site_text+second_site_text
            new_thing=new_thing+xml_contents[site_point+1:]
            xml_file=open(self.file_dir+self.filename,"w")
            xml_file.write(new_thing)
            xml_file.close()

            self._model = load_model_from_path(self.file_dir + self.filename)
            self._sim = MjSim(self._model)
            object_location=self._get_obj_size()
            states=[self._sim.data.qpos[0],self._sim.data.qpos[1],self._sim.data.qpos[2],self._sim.data.qpos[3],self._sim.data.qpos[5],self._sim.data.qpos[7],object_location[0],object_location[1],object_location[2]]
            self._set_state(np.array(states))
            self._get_trans_mat_wrist_pose()
        
    def step(self, action_list):
        """ Takes an RL timestep - conducts action for a certain number of simulation steps, indicated by frame_skip
            action: array of finger joint velocity values (finger1, finger1, finger3)
            graspnetwork: bool, set True to use grasping network to determine reward value
        """
        total_reward = 0
        self._get_trans_mat_wrist_pose()
        error = [0, 0, 0]
        
        error = self.starting_coords - self.wrist_pose
        kp = 50
        mass = 0.733
        gear = 25
        slider_motion = np.matmul(self.Tfw[0:3, 0:3], [kp * error[0], kp * error[1], kp * error[2] + mass * 10 / gear])
        slider_motion[0] = -slider_motion[0]
        slider_motion[1] = -slider_motion[1]

        self.prev_pose = np.copy(self.wrist_pose)
        for _ in range(self.frame_skip):
            if self.step_coords == 'global':
                slide_vector = np.matmul(self.Tfw[0:3, 0:3], action[0:3])
                if (self.orientation == 'rotated') & (action[2] <= 0):
                    slide_vector = [-slide_vector[0], -slide_vector[1], slide_vector[2]]
                else:
                    slide_vector = [-slide_vector[0], -slide_vector[1], slide_vector[2]]
            else:
                if (self.orientation == 'rotated') & (action[2] <= 0):
                    slide_vector = [-slide_vector[0], -slide_vector[1], slide_vector[2]]
                else:
                    slide_vector = [-action[0], -action[1], action[2]]
            for i in range(3):
                self._sim.data.ctrl[(i) * 2] = slide_vector[i]
                if self.step_coords == 'rotated':
                    self._sim.data.ctrl[i + 6] = action[i + 3] + 0.05
                else:
                    self._sim.data.ctrl[i + 6] = action[i + 3]
                self._sim.data.ctrl[i * 2 + 1] = stuff[i]
            self._sim.step()

        else:
            for _ in range(self.frame_skip):
                joint_velocities = action[0:7]
                finger_velocities = action[7:]
                for i in range(len(joint_velocities)):
                    self._sim.data.ctrl[i + 10] = joint_velocities[i]
                for i in range(len(finger_velocities)):
                    self._sim.data.ctrl[i + 7] = finger_velocities[i]
                self._sim.step()
        obs = self._get_obs()
        self._get_trans_mat_wrist_pose()
        if self.starting_coords[0] == 10:
            self.starting_coords = np.copy(self.wrist_pose)
        if not graspnetwork:
            total_reward, info, done = self._get_reward(self.with_grasp_reward)
        else:
            ### Get this reward for grasp classifier collection ###
            total_reward, info, done = self._get_reward_DataCollection()
        return obs, total_reward, done, info