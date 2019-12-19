# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:06:47 2019

@author: simon
"""
from scipy.integrate import solve_ivp
import numpy as np
import glob
import copy

def f_minus_gx_1(t, x):
    x0 = -2*x[0]*x[1] + 3*x[2]
    x1 = -2*x[0]*x[1] + 3*x[2]
    x2 = 2*x[0]*x[1] - 3*x[2]
    
    sol = np.array([x0,x1,x2])
    
    return sol

def f_over_g_minus_x_1(t, x):
    
    index = np.where(x<=0.01)[0]
    x[index] = 0.01
    
    x0 =  3*x[2]/(2*x[1]) - x[0]
    x1 = 3*x[2]/(2*x[0]) - x[1]
    x2 = 2*x[0]*x[1]/3 - x[2]
    
    sol = np.array([x0,x1,x2])
    
    return sol


def f_minus_gx_2(t, x):
    x0 = -2*x[0]*x[1] + 5*x[2]
    x1 = -2*x[0]*x[1] + 5*x[2]
    x2 = 2*x[0]*x[1] - 5*x[2]
    
    sol = np.array([x0,x1,x2])
    
    return sol

def f_over_g_minus_x_2(t, x):
    
    index = np.where(x<=0.01)[0]
    x[index] = 0.01
    
    x0 =  5*x[2]/(2*x[1]) - x[0]
    x1 = 5*x[2]/(2*x[0]) - x[1]
    x2 = 2*x[0]*x[1]/5 - x[2]
    
    sol = np.array([x0,x1,x2])
    
    return sol


def f_minus_gx_3(t, x):
    x0 = -2*x[0]*x[1] + 3*x[2]
    x1 = -2*x[0]*x[1] + 3*x[2]
    x2 = 2*x[0]*x[1] - 3*x[2]
    
    sol = np.array([x2,x1,x0])
    
    return sol

def f_over_g_minus_x_3(t, x):
    index = np.where(x<=0.01)[0]
    x[index] = 0.01
    x0 =  3*x[2]/(2*x[1]) - x[0]
    x1 = 3*x[2]/(2*x[0]) - x[1]
    x2 = 2*x[0]*x[1]/3 - x[2]
    
    sol = np.array([x2,x1,x0])
    
    return sol

def f_minus_gx_4(t, x):
    x0 = -2*x[0]*x[1] + 3*x[2]
    x1 = -2*x[0]*x[1] + 3*x[2]
    x2 = 2*x[0]*x[1] - 3*x[2]
    
    sol = np.array([x0,x2,x1])
    
    return sol

def f_over_g_minus_x_4(t, x):
    
    index = np.where(x<=0.01)[0]
    x[index] = 0.01    

    x0 =  3*x[2]/(2*x[1]) - x[0]
    x1 = 3*x[2]/(2*x[0]) - x[1]
    x2 = 2*x[0]*x[1]/3 - x[2]

    sol = np.array([x0,x2,x1])
    
    return sol


def f_minus_gx_5(t, x):
    x0 = -2*x[0]*x[1] + 3*x[2]*x[0] + 0.1
    x1 = -2*x[0]*x[1] + 3*x[2]*x[0]
    x2 = 2*x[0]*x[1] - 3*x[2]*x[0]
    
    sol = np.array([x0,x1,x2])
    
    return sol

def f_over_g_minus_x_5(t, x):
    
    index = np.where(x<=0.01)[0]
    x[index] = 0.01
    #print(x)
    x0 =  0.1/(2*x[1] - 3*x[2]) - x[0]
    x1 = 3*x[2]/2 - x[1]
    x2 = 2*x[1]/3 - x[2]

    sol = np.array([x0,x1,x2])
    
    return sol


def generateXY(num_variables = 3, constant_low=0, constant_high=100, t_span = np.linspace(0, 10000, 100*10000)):
    
    success = False
    x = 0
    y = 0
    while not success:
    
        initial_conditions = np.random.uniform(constant_low, constant_high, num_variables)#[0]
    #initial_conditions =initial_conditions.reshape(1,3)
        x = solve_ivp(f_over_g_minus_x_5, t_span = [0, 10000], y0=initial_conditions, method = 'Radau', t_eval = t_span, max_step = 100000000)   
        y = solve_ivp(f_minus_gx_5, t_span = [0, 10000], y0=initial_conditions, method = 'Radau', t_eval = t_span)
        
        success = x.success
    
        x = x.y
    
        x = x/x.max()
        x = x/np.linalg.norm(x, ord = 1, axis = 0)
        y = y.y[:,-1]
        y = y/y.max()
        y = y/np.linalg.norm(y, ord = 1)
    return x, y

def generateXY2(num_variables = 3, constant_low=0, constant_high=100, t_span = np.linspace(0, 10000, 1*10000)):
    
    success = False
    x = 0
    y = 0
    while not success:    

        initial_conditions = np.random.uniform(constant_low, constant_high, num_variables)#[0]
        #initial_conditions =initial_conditions.reshape(1,3)
        x = solve_ivp(f_over_g_minus_x_4, t_span = [0, 10000], y0=initial_conditions, method = 'LSODA', t_eval = t_span)   
        y = solve_ivp(f_minus_gx_4, t_span = [0, 10000], y0=initial_conditions, method = 'Radau', t_eval = t_span)
        
        success = x.success
        
        x = x.y
        
        x = x/x.max()
        x = x/np.linalg.norm(x, ord = 1, axis = 0)
        y = y.y[:,-1]
        y = y/y.max()
        y = y/np.linalg.norm(y, ord = 1)
    return x, y


def generateXY3(num_variables = 3, constant_low=0, constant_high=100, t_span = np.linspace(0, 10000, 1*10000)):
    
    success = False
    x = 0
    y = 0
    while not success:  
    
        initial_conditions = np.random.uniform(constant_low, constant_high, num_variables)#[0]
        #initial_conditions =initial_conditions.reshape(1,3)
        x = solve_ivp(f_over_g_minus_x_3, t_span = [0, 10000], y0=initial_conditions, method = 'LSODA', t_eval = t_span)   
        y = solve_ivp(f_minus_gx_3, t_span = [0, 10000], y0=initial_conditions, method = 'Radau', t_eval = t_span)
        
        success = x.success
        
        x = x.y
        
        x = x/x.max()
        x = x/np.linalg.norm(x, ord = 1, axis = 0)
        y = y.y[:,-1]
        y = y/y.max()
        y = y/np.linalg.norm(y, ord = 1)
    return x, y


def generateXY4(num_variables = 3, constant_low=0, constant_high=100, t_span = np.linspace(0, 10000, 1*10000)):
    success = False
    x = 0
    y = 0
    while not success:  
        initial_conditions = np.random.uniform(constant_low, constant_high, num_variables)#[0]
    #initial_conditions =initial_conditions.reshape(1,3)
        x = solve_ivp(f_over_g_minus_x_2, t_span = [0, 10000], y0=initial_conditions, method = 'RK45', t_eval = t_span)   
        y = solve_ivp(f_minus_gx_2, t_span = [0, 10000], y0=initial_conditions, method = 'Radau', t_eval = t_span)
        success = x.success
        x = x.y
        
        x = x/x.max()
        x = x/np.linalg.norm(x, ord = 1, axis = 0)
        y = y.y[:,-1]
        y = y/y.max()
        y = y/np.linalg.norm(y, ord = 1)
    return x, y


def generateXY5(num_variables = 3, constant_low=0, constant_high=100, t_span = np.linspace(0, 10000, 1*10000)):
    success = False
    x = 0
    y = 0
    while not success:  
        
        initial_conditions = np.random.uniform(constant_low, constant_high, num_variables)#[0]
        #initial_conditions =initial_conditions.reshape(1,3)
        x = solve_ivp(f_over_g_minus_x_1, t_span = [0, 10000], y0=initial_conditions, method = 'RK45', t_eval = t_span)   
        y = solve_ivp(f_minus_gx_1, t_span = [0, 10000], y0=initial_conditions, method = 'Radau', t_eval = t_span)
        success = x.success
        x = x.y
        
        x = x/x.max()
        x = x/np.linalg.norm(x, ord = 1, axis = 0)
        y = y.y[:,-1]
        y = y/y.max()
        y = y/np.linalg.norm(y, ord = 1)
    return x, y





X = []
Y = []

for i in range(0,2000):
    
    x,y = generateXY2()
    X.append(x.T)
    Y.append(y)
    
    x,y = generateXY2()
    X.append(x.T)
    Y.append(y)
    
    x,y = generateXY3()
    X.append(x.T)
    Y.append(y)
    
    x,y = generateXY4()
    X.append(x.T)
    Y.append(y)
 
    x,y = generateXY5()
    X.append(x.T)
    Y.append(y)    
    '''
    if i == 0:
        X = x.T
        Y = y
    else:
        X = np.append(X, x.T,  axis=0)
        Y = np.append(Y,y, axis=0)
        
    x,y = generateXY2()
    X = np.append(X, x.T,  axis=0)
    Y = np.append(Y,y, axis=0)
    print(i)
 
    x,y = generateXY3()
    X = np.append(X, x.T,  axis=0)
    Y = np.append(Y,y, axis=0)
    print(i)
    
    x,y = generateXY4()
    X = np.append(X, x.T,  axis=0)
    Y = np.append(Y,y, axis=0)
    print(i)
    
    x,y = generateXY5()
    X = np.append(X, x.T,  axis=0)
    Y = np.append(Y,y, axis=0)
    '''
    print(i)
    
    
    
X = np.array(X)
Y = np.array(Y)
np.save('X.npy', X)
np.save('Y.npy', Y)


    