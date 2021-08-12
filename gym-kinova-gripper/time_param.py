#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:33:29 2021

@author: orochi
"""
import os
from time_param_base import TimeParamBase

class TimeParamMujoco(TimeParamBase):
    def __init__(self, xml_path, json_path = os.path.dirname(__file__)+'/config/time.json'):
        super().__init__(json_path)
    
    def write_params(self,filepath):
        """this function writes the timestep parameters to the apropriate file
        to update the simulator"""
        xml_file=open(filepath,"r")
        xml_contents=xml_file.read()
        xml_file.close()
        a=xml_contents.find('<option timestep')
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