# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:43:47 2019

@author: simon
"""
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import LSTM,Dense, TimeDistributed, Reshape, Lambda
from keras.layers import multiply
import keras




def solverModel(in_shape, outshape):
    print(in_shape)
    Input1 = Input(in_shape)
    net = Input1
    net = LSTM(3, activation = 'relu',  return_sequences = True)(Input1)
    net = LSTM(3, activation = 'relu')(net)
    net = Dense(8, activation = 'relu')(net)
    net = Dense(8, activation = 'relu')(net)
    out = Dense(outshape, activation = 'softmax')(net)
    model = Model(inputs=Input1,outputs=out)
    return model


X=np.load('X.npy')
Y=np.load('Y.npy')





solver = solverModel((1000000,3), 3)
solver.summary()
solver.compile(loss='mse', optimizer='adam', metrics=['mse'])


solver.fit(X, Y, epochs=150, batch_size=1)




