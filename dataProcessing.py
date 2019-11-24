# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:43:25 2019

@author: simon
"""
import numpy as np
import glob

path = './DataNormal/data*.npy'
files = glob.glob(path)


steadyStateData = []
for file in files:
    data = np.load(file)
    steadyStateData.append(data[-1,:])


steadyStateData = np.array(steadyStateData)
np.save("dataNormal", steadyStateData)


path = './DataAlter/data*.npy'
files = glob.glob(path)


steadyStateData = []
for file in files:
    data = np.load(file)
    steadyStateData.append(data[-1,:])


steadyStateData = np.array(steadyStateData)
np.save("dataAlter", steadyStateData)