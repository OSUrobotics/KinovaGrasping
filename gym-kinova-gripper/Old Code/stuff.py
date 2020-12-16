#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:59:13 2019

@author: orochi
"""
import numpy as np
import csv
from classifier_network import LinearNetwork
from classifier_network import ReducedLinearNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def calc_velocity(start,end):
    delta_t=0.05
    #print(type(start),type(end))
    velocity=(end-start)/delta_t    
    return velocity

def normalize_vector(vector):
    #print(vector-np.min(vector))
    #print(np.max(vector)-np.min(vector))
    if (np.max(vector)-np.min(vector)) == 0:
        n_vector=np.ones(np.shape(vector))*0.5
    else:
        n_vector=(vector-np.min(vector))/(np.max(vector)-np.min(vector))
    return n_vector


filenames=['Classifier_Data_Big_Cube.csv','Classifier_Data_Med_Cube.csv','Classifier_Data_Small_Cube.csv', \
           'Classifier_Data_Big_Cylinder.csv','Classifier_Data_Med_Cylinder.csv','Classifier_Data_Small_Cylinder.csv']
a=[]
column_names=[]

#load in the data to one massive matrix called a
for k in range(6):
    with open('Classifier_Data/'+filenames[k]) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                column_names.append(row)
                #print(f'Column names are {", ".join(row)}')
                #print(row[6],row[48])
                line_count += 1
            else:
                a.append(row)
                line_count += 1
            #print('here')
        #print(f'Processed {line_count} lines.')
    #print(np.shape(a))
network=ReducedLinearNetwork()
network.zero_grad()
network.double()

b=np.shape(a)
print(b)

a=np.array(a,dtype='f')

#create a list of numbers that correspond to the columns to be removed. This arrangement removes the roll, pitch and yaw from the matrix a
c=np.arange(9,42,6)
d=np.arange(10,42,6)
e=np.arange(11,42,6)
f=np.arange(51,87,6)
g=np.arange(52,87,6)
h=np.arange(53,87,6)
#obj_pose=np.array([84,85,86])
c=np.concatenate((c,d,e,f,g,h))

#calculate the velocity of the fingers. 
for i in range(36):
    velocity=calc_velocity(a[:,i+6],a[:,i+48])
    a[:,i+6]=velocity
#normalize the entire table so that all the inputs and outputs lie on a spectrum from 0-1
for i in range(b[1]):
    a[:,i]=normalize_vector(a[:,i])
#remove the columns that are unwanted, described by the array c
new_a=np.zeros([b[0],69])
for i in range(b[0]):
    new_a[i,:]=np.delete(a[i,:],c)
#check to make sure the right columns got deleted
column_names=np.delete(column_names,c)
print(column_names[0])
a=new_a
#print(a[:,-1])
running_loss=0
learning_rate=0.1
total_loss=[]
total_time=[]
num_epocs=100
network= network.float()
for j in range(num_epocs):
    print(j)
    learning_rate=0.1-j/num_epocs*0.09
    np.random.shuffle(a)
    running_loss=0
    for i in range(b[0]):
        #network=network.float()
        #state = ego.convert_world_state_to_front() 
        #ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, network)
        input1=a[i,:-1]
        #print(input1)
        network_input=torch.tensor(input1)
        #print(network_input)
        #print(a[i,-1])
        network_target=torch.tensor(a[i,-1])
        #network_target.reshape(1)
        
        network_input=network_input.float()
        #print(network_input)
        out=network(network_input)
        out.reshape(1)
        network.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(out, network_target)
        loss.backward()
        running_loss += loss.item()
        #print(out.data,network_target.data, out.data-network_target.data)
        #print(loss.item())
        for f in network.parameters():
            f.data.sub_(f.grad.data * learning_rate)
        if i % 1000 ==999: # keep a tally of the loss and time so that the training can be plotted
            print(running_loss)
            #print(loss.item(),out[0])
            total_loss.append(running_loss)
            total_time.append((i+1)/1000+j*b[0]/1000)
            running_loss=0
plt.plot(total_time,total_loss)
plt.show()
torch.save(network.state_dict(),'./full_trained_classifier_no_rpw_obj_pose.pth')