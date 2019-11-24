# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:43:25 2019

@author: simon
"""

path = './DataNormal/*.npy'
files = glob.glob(path)


steadyStateData = []
for file in files:
    data = np.load(file)
    steadyStateData.append(data[-1,:])


steadyStateData = np.array(steadyStateData)
np.save("dataNormal", steadyStateData)


path = './DataAlter/*.npy'
files = glob.glob(path)


steadyStateData = []
for file in files:
    data = np.load(file)
    steadyStateData.append(data[-1,:])


steadyStateData = np.array(steadyStateData)
np.save("dataAlter", steadyStateData)