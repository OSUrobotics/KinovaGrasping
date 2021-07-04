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