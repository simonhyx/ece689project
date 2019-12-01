# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:38:50 2019

@author: simon
"""


from dataGeneratorEdge_redo_v2 import dataGenerator
import numpy as np

from scipy.integrate import odeint

generator = dataGenerator()










for i in range(0,100):
    print(i)
    name = "./DataNormal/" + 'data_' + str(i)
    init_name =  "./DataNormal/" + 'init_cond_' + str(i)
    #sol = generator.generateData((0,24*3600), init)
    
    init = generator.sampleInitialCond(generator.custom_ranges)
    
    time = [0, 24*3600, 100*24*3600]
    
    sol = generator.generateDatav2(time = time, initial_cond = init)
    
    downSampled = []
    for i in range(0,100*24*3600, 3600):
        downSampled.append(i)
    sol = sol[downSampled,:]
    np.save(name, sol)
    np.save(init_name, np.array(init))
    
    
for i in range(0,100):
    print(i)
    name = "./DataAlter/" + 'data_' + str(i)
    init_name =  "./DataAlter/" + 'init_cond_' + str(i)
    #sol = generator.generateData((0,24*3600), init)
    
    init = generator.sampleInitialCond(generator.custom_ranges2)
    
    time = [0, 24*3600, 100*24*3600]
    
    sol = generator.generateDatav2(time = time, initial_cond = init)
    
    downSampled = []
    for i in range(0,100*24*3600, 3600):
        downSampled.append(i)
    sol = sol[downSampled,:]
    np.save(name, sol)
    np.save(init_name, np.array(init))
