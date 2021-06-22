#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:17:47 2021

@author: orochi
"""

class Controller():
    
    def __init__(self,controller_type='naive'):
        self.controller_type = controller_type
        
    def calc_action(self):
        return eval('self.' + self.controller_type + '()')
    
    def naive(self):
        return [0, 0, 0, 0.5, 0.5, 0.5]
    
    def expert(self):
        
    def RL(self):
        