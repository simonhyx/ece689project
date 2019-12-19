# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:59:21 2019

@author: simon
"""

from model.skyNet5_nodeMerge import T800Node, SkyNet
from model.dataGeneratorEdge_redo_v2 import dataGenerator

from scipy.integrate import solve_ivp
import numpy as np
import glob
import copy

from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import LSTM,Dense, TimeDistributed, Reshape, Lambda
from keras.layers import multiply
import keras




def FandGModel(in_size, outshape):

    Initial_conditions = Input(in_size)
    f = Dense(8, activation = 'relu')(Initial_conditions)
    f = Dense(8, activation = 'relu')(f)
    f = Dense(outshape, activation = 'relu')(f)
    
    g = Dense(8, activation = 'relu')(Initial_conditions)
    g = Dense(8, activation = 'relu')(g)
    g = Dense(outshape, activation = 'relu')(g)
    
    gx = multiply([g, Initial_conditions])
    f_minus_gx = keras.layers.Subtract()([f, gx])
    out1 = Activation('linear')(f_minus_gx)
    
    g_offset = Input(in_size)
    g_plus_bias = keras.layers.Add()([g, g_offset])
    f_over_g = Lambda(lambda x: x[0]/x[1])([f, g_plus_bias])
    f_over_g_minus_x = keras.layers.Subtract()([f_over_g, Initial_conditions])
    out2 = Activation('linear')(f_over_g_minus_x)
    
    model = Model(inputs=[Initial_conditions, g_offset],outputs=[out1, out2])

    return model


def solverModel(in_shape, outshape):
    print(in_shape)
    Input1 = Input(in_shape)
    net = Input1
    net = LSTM(8, activation = 'relu',  return_sequences = True)(Input1)
    net = LSTM(8, activation = 'relu')(net)
    net = Dense(8, activation = 'relu')(net)
    net = Dense(8, activation = 'relu')(net)
    out = Dense(outshape, activation = 'softmax')(net)
    model = Model(inputs=Input1,outputs=out)
    return model

f_and_g = FandGModel((3,), 3)
f_and_g.summary()
g_offset = np.ones(3)


def numericIntegrate_f_minus_gx(t, x):
    global f_and_g
    global g_offset
    index = np.where(np.isnan(x))[0]
    x[index] = 10**37
    
    #print(x)
    #wtf = [x, g_offset.reshape(1,3)]
    [f_minus_gx, f_over_g_minus_x] = f_and_g.predict([x.reshape(1,x.shape[0]), g_offset.reshape(1,g_offset.shape[0])])
    #print(f_minus_gx.reshape(x.shape[0]))
    
    index = np.where(np.isnan(f_minus_gx))[0]
    f_minus_gx[index] = 10**37
    
    return f_minus_gx.reshape(x.shape[0])
    
def numericIntegrate_f_over_g_minus_x(t, x):
    global f_and_g
    global g_offset
    
    index = np.where(np.isnan(x))[0]
    x[index] = 10**37
    [f_minus_gx, f_over_g_minus_x] = f_and_g.predict([x.reshape(1,x.shape[0]), g_offset.reshape(1,g_offset.shape[0])])
    
    index = np.where(np.isnan(f_over_g_minus_x))[0]
    f_over_g_minus_x[index] = 10**37    
    
    return f_over_g_minus_x.reshape(x.shape[0])


def generateXY(num_variables = 3, constant_low=0, constant_high=100, t_span = np.linspace(0, 10000, 100*10000)):
    initial_conditions = np.random.uniform(constant_low, constant_high, num_variables)#[0]
    #initial_conditions =initial_conditions.reshape(1,3)
    x = solve_ivp(numericIntegrate_f_minus_gx, t_span = [0, 10000], y0=initial_conditions, method = 'Radau', t_eval = t_span)
    y = solve_ivp(numericIntegrate_f_over_g_minus_x, t_span = [0, 10000], y0=initial_conditions, method = 'Radau', t_eval = t_span)
    
    x = x.y
    
    x = x/x.max()
    x = x/np.linalg.norm(x, ord = 1, axis = 0)
    y = y.y[:,-1]
    y = y/y.max()
    y = y/np.linalg.norm(y, ord = 1)
    return x, y


X = []
Y = []

for i in range(0,2):
    x,y = generateXY()
    X.append(x.T)
    Y.append(y)
    
X = np.array(X)
Y = np.array(Y)
np.save('X.npy', X)
np.save('Y.npy', Y)




solver = solverModel((1000000,3), 3)
solver.summary()
solver.compile(loss='mse', optimizer='adam', metrics=['mse'])


solver.fit(X, Y, epochs=150, batch_size=1)
'''
def generatorModel(in_shape, outshape):
    print(in_shape)
    Input1 = Input(in_shape)
    net = Input1
    net = Dense(8, activation = 'relu')(Input1)
    net = Dense(8, activation = 'relu')(net)
    net = Dense(8, activation = 'relu')(net)
    out = Dense(outshape, activation = 'softmax')(net)
    model = Model(inputs=Input1,outputs=out)
    return model


def diffEq(t, x):
    global problemGenerator
    sol = problemGenerator.predict(x)
    return sol

def generateSamples(input_size, constant_low, constant_high, t_span = np.linspace(0, 10000, 100*10000)):
    # This function is used to generate datapoints
    # Each datapoint is a time series data generated from a numeric integration of the generator function created in generatorModel
    # The generator Model is an RL agent that takes the current state X as input and outputs the rate of change dX/dt which is a function 
    # that is generated my the generator model. This model generates the two functions f and g simultaneously, and we use the outputs of f and g
    # at each step to compute the true evolution of dX/dt = f - gx and dX/dt = f/g - x
    # The solver is a recurrent neural network model that takes the predicted dX/dt = f/g - x values as impute and tries to predict 
    # the final states of dX/dt = f - gx at steady state. 
    
    
    
    # start with numeric integration of neural networks
    
    # generate random initial conditions
    initial_conditions = np.random.uniform(constant_low, constant_high, 1)[0]
    x = solve_ivp(f_minus_xg_model, t_span = [0, 10000], y0=initial_conditions, method = 'Radau', t_eval = t_span)
    y = solve_ivp(f_over_g_minus_x_model, t_span = [0, 10000], y0=initial_conditions, method = 'Radau', t_eval = t_span)
    
    return x, y

problemGenerator_f = generatorModel()
problemGenerator_p = generatorModel()
problemSolver = solverModel()

def f_minus_xg_model(t,x):
    global globalConst1
    global globalPoly1
    global globalConst2
    global globalPoly2
    
    p1 = []
    p2 = []
    for i in range(0, len(x)):
        p1.append( polnomial(globalConst1[i], x, globalPoly1[i]) )
        p2.append( polnomial(globalConst2[i], x, globalPoly2[i]) )
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    return p1-x*p2

def f_over_g_minus_x_model(t,x):
    global globalConst1
    global globalPoly1
    global globalConst2
    global globalPoly2
    
    p1 = []
    p2 = []
    for i in range(0, len(x)):
        p1.append( polnomial(globalConst1[i], x, globalPoly1[i]) )
        p2.append( polnomial(globalConst2[i], x, globalPoly2[i]) )
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    return p1/p2-x

globalConst1=0
globalPoly1=0

globalConst2=0
globalPoly2=0


def test():
    global globalConst1
    print(globalConst1)
    globalConst1 = 5

def polnomial(constants, variables, degrees):
    return np.sum(constants*np.prod(variables**degrees , axis=1))
#np.multiply(variables**degrees, axis=1)
def sampleOnePolynomial(constant_low, constant_high, num_terms, degree_min, degree_max, variables):
    c=[] #constants
    p=[] #degrees 
    
    for i in range(0,num_terms):
        c.append(np.random.uniform(constant_low, constant_high, 1)[0])
        p.append(np.random.randint(degree_min, degree_max, len(variables)))
        
    c = np.array(c)
    p = np.array(p)
    
    return c, p




def f_minus_xg(t,x):
    global globalConst1
    global globalPoly1
    global globalConst2
    global globalPoly2
    
    p1 = []
    p2 = []
    for i in range(0, len(x)):
        p1.append( polnomial(globalConst1[i], x, globalPoly1[i]) )
        p2.append( polnomial(globalConst2[i], x, globalPoly2[i]) )
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    return p1-x*p2

def f_over_g_minus_x(t,x):
    global globalConst1
    global globalPoly1
    global globalConst2
    global globalPoly2
    
    p1 = []
    p2 = []
    for i in range(0, len(x)):
        p1.append( polnomial(globalConst1[i], x, globalPoly1[i]) )
        p2.append( polnomial(globalConst2[i], x, globalPoly2[i]) )
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    return p1/p2-x



def simulation(constant_low, constant_high, num_terms, degree_min, degree_max, variables, t_span = np.linspace(0, 10000, 100*10000)):
    global globalConst1
    global globalPoly1
    global globalConst2
    global globalPoly2
    
    globalConst1 = []
    globalPoly1 = []
    globalConst2 = []
    globalPoly2 = []
    
    for i in range(0, len(variables)):
    
        c,p = sampleOnePolynomial(constant_low, constant_high, num_terms, degree_min, degree_max, variables)
        globalConst1.append(c)
        globalPoly1.append(p)
        
        c,p = sampleOnePolynomial(constant_low, constant_high, num_terms, degree_min, degree_max, variables)
        globalConst2.append(c)
        globalPoly2.append(p)
        
    globalConst1 = np.array(globalConst1)
    globalPoly1 = np.array(globalPoly1)
    globalConst2 = np.array(globalConst2)
    globalPoly2 = np.array(globalPoly2)   
    
    x = solve_ivp(f_minus_xg, t_span = [0, 10000], y0=variables, method = 'Radau', t_eval = t_span)
    y = solve_ivp(f_over_g_minus_x, t_span = [0, 10000], y0=variables, method = 'Radau', t_eval = t_span)
    
    return x, y





def sampleGenerator(num_terms=3):
    variables = np.random.uniform(0, 1000, num_terms)

    #f_minus_xg(3,variables)
    #f_over_g_minus_x(3, variables)

    x,y = simulation(0,10, num_terms, 0, 2, variables)

    x=x.y[:,-1]
    y=y.y[:,-1]
    return x,y

sampleGenerator()


def myModel(in_shape, outshape):
    print(in_shape)
    Input1 = Input(in_shape)
    net = Input1
    net = LSTM(8, activation = 'relu',  return_sequences = True)(Input1)
    net = LSTM(8, activation = 'relu')(net)
    net = Dense(8, activation = 'relu')(net)
    net = Dense(8, activation = 'relu')(net)
    out = Dense(outshape, activation = 'softmax')(net)
    model = Model(inputs=Input1,outputs=out)
    return model


t_span = np.linspace(0, 10000, 100*10000)

model = myModel((len(t_span),3),3)

'''



'''

in_shape = (3,len(t_span))
outshape = 3

sol = solve_ivp(func1, t_span = [0, 100000], y0=[1000, 1000, 1000])


sol2 = solve_ivp(func2, t_span = [0, 100000], y0=np.array([1000, 1000, 1000]))


x_in = np.array([7000, 3000, 1000])
#x_in = x_in/np.linalg.norm(x_in, ord = 1)


sol3 = solve_ivp(func3, t_span = [0, 10000], y0=x_in)
y3 = sol3.y[:,-1]
y3/sum(y3)

#y3 = sol3.y[-1]


sol4 = solve_ivp(func4, t_span = [0, 10000], y0=x_in)
y4= sol4.y[:,-1]
y4/sum(y4)

sol5 = solve_ivp(func5, t_span = [0, 10000], y0=x_in, method = 'LSODA')
y5= sol5.y[:,-1]
y5/sum(y5)

y5/np.linalg.norm(y5, ord = 1)

#y4 = sol4.y[-1]


'''