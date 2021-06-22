#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:16:39 2021

@author: orochi
"""

class Angle(State):
    def __init__(self,sim):
        print('intialized angle')
        self._sim=sim