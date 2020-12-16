#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:36:32 2020

@author: orochi
"""

import numpy as np
import climin
import GPy
from ipywidgets import Text
from IPython.display import display
import sys
import torch
import datetime
    
def trainGP(all_training_set,all_training_label,all_testing_set,all_testing_label):
    t = Text(align='right')
    display(t)
    batchsize = 10
    Z = np.random.rand(20,72)
    all_training_label=np.vstack(all_training_label)
    m = GPy.core.SVGP(all_training_set, all_training_label, Z, GPy.kern.RBF(72) + GPy.kern.White(72), GPy.likelihoods.Gaussian(), batchsize=batchsize)
    m.kern.white.variance = 1e-5
    m.kern.white.fix()
    
    opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.2, momentum=0.9)
    def callback(i):
        t.value = str(m.log_likelihood())
        #Stop after 288615 iterations
        if i['n_iter'] > 100000:
            return True
        return False
    
    info = opt.minimize_until(callback)
    all_answers=m.predict(all_testing_set)
    
    answer_shape=np.shape(all_answers)
    percent_right=np.zeros(answer_shape[1])
    for i in range(answer_shape[1]):
        if all_answers[0][i]>0.5:
            percent_right[i]=1
        else:
            percent_right[i]=0
    final_percent=np.sum(abs(all_testing_label-percent_right))/answer_shape[1]
    print('classifier got ', 1-final_percent*100, '% correct on the test set')
    print("Finish training, saving...")
    

    # 1: Saving a model:
    np.save('model_save.npy', m.param_array)
    # 2: loading a model
    # Model creation, without initialization:
    #m_load = GPy.models.GPRegression(X, Y, initialize=False)
    #m_load.update_model(False) # do not call the underlying expensive algebra on load
    #m_load.initialize_parameter() # Initialize the parameters (connect the parameters up)
    #m_load[:] = np.load('model_save.npy') # Load the parameters
    #m_load.update_model(True) # Call the algebra only once
    #print(m_load)
    model_path="trained_model"
    np.save(model_path + "_" + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + ".npy",m.param_array)  
    return m

def load_GP(filepath,training_set,training_label):
    Z = np.random.rand(20,72)
    training_label=np.vstack(training_label)
    m_load = GPy.core.SVGP(training_set, training_label, Z, GPy.kern.RBF(72) + GPy.kern.White(72), GPy.likelihoods.Gaussian(), batchsize=10,initialize=False)
    m_load.update_model(False)
    m_load.initialize_parameter()
    m_load[:] = np.load(filepath)
    m_load.update_model(True)
    return m_load

def test_GP(testing_set,testing_label,model):
    num_labels=[200,200,200,200,200,200]
    percent_correct=[0,0,0,0,0,0]
    all_answers=model.predict(testing_set)
    print(all_answers[0][:])
    answer_shape=np.shape(all_answers)
    percent_right=np.zeros(answer_shape[1])
    past_vals=0
    for i in range(3):
        for j in range(num_labels[i]):
            if all_answers[0][j+past_vals]>0.5:
                percent_right[j+past_vals]=1
            else:
                percent_right[j+past_vals]=0
        percent_correct[i]=1-np.sum(abs(testing_label[past_vals:j+past_vals]-percent_right[past_vals:j+past_vals]))/num_labels[i]
        past_vals=past_vals+num_labels[i]
        print(past_vals)
        print('correctly identified grasps from set',i,percent_correct[i]*100, '% of the time')
    final_percent=1-np.sum(abs(testing_label-percent_right))/answer_shape[1]
    print('classifier got ', final_percent*100, '% correct on the test set')
    return final_percent