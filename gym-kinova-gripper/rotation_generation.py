#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:07:06 2020

@author: orochi
"""
import numpy as np
top_rotation=np.zeros([5000,3])
side_rotation=np.zeros([5000,3])
norm_rotation=np.zeros([5000,3])
top_text=np.empty([5000],dtype=object)
norm_text=np.empty([5000],dtype=object)
side_text=np.empty([5000],dtype=object)
#shape_names=["CubeS","CubeM","CubeB","CylinderS","CylinderM","CylinderB","Cube45S","Cube45M","Cube45B","Vase1S","Vase1M","Vase1B","Vase2S","Vase2M","Vase2B",\
 #            "Cone1S","Cone1M","Cone1B","Cone2S","Cone2M","Cone2B"]
shape_names=["LemonS","LemonM","LemonB","RbowlS","RbowlM","RbowlB","RectbowlS","RectbowlM","RectbowlB","BottleS","BottleM","BottleB","TBottleS","TBottleM","TBottleB"]
for name in shape_names:
    for i in range(5000):
        top_rotation[i,:]=np.random.normal(-0.087,0.087,3)
        norm_rotation[i,:]=np.random.normal(-0.087,0.087,3)
        side_rotation[i,:]=np.random.normal(-0.087,0.087,3)
        top_rotation[i,:]=np.array([-1.57,0,-1.57])+top_rotation[i,:]
        norm_rotation[i,:]=np.array([0,0,0])+norm_rotation[i,:]
        side_rotation[i,:]=np.array([-1.2,0,0])+side_rotation[i,:]
        temp=np.array2string(top_rotation[i,:],separator=' ')
        top_text[i]=temp[1:-1]+"\n"
        temp=np.array2string(norm_rotation[i,:],separator=' ')
        norm_text[i]=temp[1:-1]+"\n"
        temp=np.array2string(side_rotation[i,:],separator=' ')
        side_text[i]=temp[1:-1]+"\n"
    file1 = open(name+"_top_rotation.txt","w+")
    file1.writelines(top_text)
    file1.close()
    file2 = open(name+"_side_rotation.txt","w+")
    file2.writelines(side_text)
    file2.close()
    file3 = open(name+"_norm_rotation.txt","w+")
    file3.writelines(norm_text)
    file3.close()

'''

        all_objects[] =  "/kinova_description/j2s7s300_end_effector_v1_bhg.xml"
        all_objects["HourM"] =  "/kinova_description/j2s7s300_end_effector_v1_mhg.xml"
        all_objects["HourS"] =  "/kinova_description/j2s7s300_end_effector_v1_shg.xml"
        # Vase
        all_objects["VaseB"] =  "/kinova_description/j2s7s300_end_effector_v1_bvase.xml"
        all_objects["VaseM"] =  "/kinova_description/j2s7s300_end_effector_v1_mvase.xml"
        all_objects["VaseS"] =  "/kinova_description/j2s7s300_end_effector_v1_svase.xml"
        # Bottle
        all_objects["BottleB"] =  "/kinova_description/j2s7s300_end_effector_v1_bbottle.xml"
        all_objects["BottleM"] =  "/kinova_description/j2s7s300_end_effector_v1_mbottle.xml"
        all_objects["BottleS"] =  "/kinova_description/j2s7s300_end_effector_v1_sbottle.xml"
        # Bowl
        all_objects["BowlB"] =  "/kinova_description/j2s7s300_end_effector_v1_bRoundBowl.xml"
        all_objects["BowlM"] =  "/kinova_description/j2s7s300_end_effector_v1_mRoundBowl.xml"
        all_objects["BowlS"] =  "/kinova_description/j2s7s300_end_effector_v1_sRoundBowl.xml"
        # Lemon
        all_objects["LemonB"] =  "/kinova_description/j2s7s300_end_effector_v1_blemon.xml"
        all_objects["LemonM"] =  "/kinova_description/j2s7s300_end_effector_v1_mlemon.xml"
        all_objects["LemonS"] =  "/kinova_description/j2s7s300_end_effector_v1_slemon.xml"
        # TBottle
        all_objects["TBottleB"] =  "/kinova_description/j2s7s300_end_effector_v1_btbottle.xml"
        all_objects["TBottleM"] =  "/kinova_description/j2s7s300_end_effector_v1_mtbottle.xml"
        all_objects["TBottleS"] =  "/kinova_description/j2s7s300_end_effector_v1_stbottle.xml"
        # RBowl
        all_objects["RBowlB"] =  "/kinova_description/j2s7s300_end_effector_v1_bRectBowl.xml"
        all_objects["RBowlM"] =  "/kinova_description/j2s7s300_end_effector_v1_mRectBowl.xml"
        all_objects["RBowlS"] =  "/kinova_description/j2s7s300_end_effector_v1_sRectBowl.xml"
'''
