#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:57:24 2021

@author: orochi
"""
import json
import time
import numpy as np

class State_Metric():
    sim=None
    def __init__(self,subdict):
        self.data=subdict
        self.maxes=[]
        self.mins=[]
        print('initialized a general state metric')

                
    def get_value(self):
        return self.data
    
    def get_rotation(self):
        wrist_pose=np.copy(State_Metric.sim.data.get_geom_xpos('palm')) 
        Rfa=np.copy(State_Metric.sim.data.get_geom_xmat('palm'))
        temp=np.matmul(Rfa,np.array([[0,0,1],[-1,0,0],[0,-1,0]]))
        temp=np.transpose(temp)
        Tfa=np.zeros([4,4])
        Tfa[0:3,0:3]=temp
        Tfa[3,3]=1       
        local_to_world=np.zeros([4,4])
        local_to_world[0:3,0:3]=temp
        local_to_world[3,3]=1
        wrist_pose=wrist_pose+np.matmul(np.transpose(local_to_world[0:3,0:3]),[-0.009,0.048,0.0])
        local_to_world[0:3,3]=np.matmul(-(local_to_world[0:3,0:3]),np.transpose(wrist_pose))
        return local_to_world
        
class Angle(State_Metric):
    def update(self,key): # this function finds either the joint angles or the x and z angle, 
        #based on a key determined by the state group or state metric function that called it
        if key=='JointState':
            arr = []
            for i in range(len(State_Metric.sim.data.sensordata)-17):
                arr.append(State_Metric.sim.data.sensordata[i])
            arr[0]=-arr[0]
            arr[1]=-arr[1]
            return arr
        
        elif key =='X,Z':
            obj_pose = State_Metric.sim.data.get_geom_xpos("object")
            local_to_world=self.get_rotation()
            local_obj_pos=np.copy(obj_pose)
            local_obj_pos=np.append(local_obj_pos,1)
            local_obj_pos=np.matmul(local_to_world,local_obj_pos)
            obj_wrist = local_obj_pos[0:3]/np.linalg.norm(local_obj_pos[0:3])
            center_line = np.array([0,1,0])
            z_dot = np.dot(obj_wrist[0:2],center_line[0:2])
            z_angle = np.arccos(z_dot/np.linalg.norm(obj_wrist[0:2]))
            x_dot = np.dot(obj_wrist[1:3],center_line[1:3])
            x_angle = np.arccos(x_dot/np.linalg.norm(obj_wrist[1:3]))
            return [x_angle,z_angle]

class Position(State_Metric):
    def update(self,keys):
        arr=[]
        local_to_world=self.get_rotation()
        for key in keys:
            temp=State_Metric.sim.data.get_geom_xpos(key)
            temp.append(1)
            temp=np.matmul(local_to_world,temp)
            arr.append(temp[0:3])
        return arr

class Vector(State_Metric):
    def update(self,key):
        local_to_world=self.get_rotation()
        gravity=[0,0,-1]
        return np.matmul(local_to_world[0:3,0:3],gravity)

class Ratio(State_Metric):
    def update(self,keylist):
        pass
        '''
        finger_pose=np.array(finger_pose)

        s1=finger_pose[0:3]-finger_pose[6:9]
        s2=finger_pose[0:3]-finger_pose[3:6]
        #print(finger_pose)
        front_area=np.linalg.norm(np.cross(s1,s2))/2
        #print('front area',front_area)
        top1=np.linalg.norm(np.cross(finger_pose[0:3],finger_pose[9:12]))/2
        top2=np.linalg.norm(np.cross(finger_pose[9:12],finger_pose[12:15]))/2
        top3=np.linalg.norm(np.cross(finger_pose[3:6],finger_pose[12:15]))/2
        top4=np.linalg.norm(np.cross(finger_pose[6:9],finger_pose[15:18]))/2
        top5=np.linalg.norm(np.cross(finger_pose[9:12],finger_pose[15:18]))/2
        total1=top1+top2+top3
        total2=top1+top4+top5
        top_area=max(total1,total2)
        #print('front',front_area,'top',top_area)

        sites=["palm","palm_1","palm_2","palm_3","palm_4"]
        obj_pose=[]#np.zeros([5,3])
        xs=[]
        ys=[]
        zs=[]
        for i in range(len(sites)):
            temp=self._sim.data.get_site_xpos(sites[i])
            temp=np.append(temp,1)
            temp=np.matmul(self.local_to_world,temp)
            temp=temp[0:3]
            if rangedata[i] < 0.06:
                temp[1]+=rangedata[i]
                obj_pose=np.append(obj_pose,temp)
            #obj_pose[i,:]=temp
        for i in range(int(len(obj_pose)/3)):
            xs=np.append(xs,obj_pose[i*3])
            ys=np.append(ys,obj_pose[i*3+1])
            zs=np.append(zs,obj_pose[i*3+2])
        if xs ==[]:
            sensor_pose=[0.2,0.2,0.2]
        else:
            sensor_pose=[np.average(xs),np.average(ys),np.average(zs)]
        obj_size=np.copy(self._get_obj_size())
        if np.argmax(np.abs(gravity))==2:
            front_part=np.abs(obj_size[0]*obj_size[2])/front_area
            top_part=np.abs(obj_size[0]*obj_size[1])/top_area
        elif np.argmax(np.abs(gravity))==1:
            front_part=np.abs(obj_size[0]*obj_size[2])/front_area
            top_part=np.abs(obj_size[1]*obj_size[2])/top_area
        else:
            front_part=np.abs(obj_size[0]*obj_size[1])/front_area
            top_part=np.abs(obj_size[0]*obj_size[2])/top_area
        '''
class Distance(State_Metric):
    def update(self,keys):
        if keys[0] == 'Rangefinder':
            range_data = []
            for i in range(17):
                if self._sim.data.sensordata[i+len(self._sim.data.sensordata)-17]==-1:
                    a=6
                else:
                    a=self._sim.data.sensordata[i+len(self._sim.data.sensordata)-17]
                range_data.append(a)
        elif keys[0] =='FingerObject':
            obj = State_Metric.sim.data.get_geom_xpos("object")
            dists = []
            for i in keys[1:]:
                pos = State_Metric.sim.data.get_site_xpos(i)
                dist = np.absolute(pos[0:3] - obj[0:3])
                temp = np.linalg.norm(dist)
                dists.append(temp)

class State_Group(State_Metric):
    valid_state_names={'Position':Position,'Distance':Distance,'Angle':Angle,'Ratio':Ratio,'Vector':Vector}
    def __init__(self,subdict):
        super().__init__(subdict)
        for name,value in subdict.items():
            print('looking at ',name,value)
            for key in State_Metric.valid_state_names.keys():
                if key in name:
                    self.template[name]=State_Metric.valid_state_names[key]({name:value})
    
    def update(self,keys):
        arr=[]
        for name,value in self.data:
            temp=value.update()
            arr.append(temp)
        return arr
