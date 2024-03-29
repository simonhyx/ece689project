# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:39:58 2019

@author: simon
"""

from __future__ import division
import warnings

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense
import numpy as np
from rl.core import*
import random
from scipy.integrate import odeint
from dataGeneratorEdge_redo_v2 import dataGenerator
from sklearn.metrics import mean_squared_error

from scipy.integrate import solve_ivp


class controlEnv():
    def __init__(self, df, allowed_genes_to_be_perturbed, target_state, max_step = 10):
        #non permutable variables
        
        self.initialStates = df
		
        self.currentData = self.getData()
        
        self.allowed_actions = allowed_genes_to_be_perturbed
        
        self.action_space = actionSpace(len(self.allowed_actions))
        
        # negative 1 for ignore states, +1 for retained states
        self.targetState = target_state
        
        self.action_list = [] 
        
        self.obs_space = len(df.columns)*2
        
        self.max_step = max_step
        self.steps =0 

		
    def getData(self):
		#index = np.random.randint(self.nRow, size=self.stocks_per_epi)
		# for choosing stocks
        index = np.random.choice(self.initialStates.shape[0], 1)[0]
		# need to randomize self.index
        #print(index)
        return self.initialStates.iloc[index:(index+1)]
	
	
	
    def action_space(self):
        return self.action_space
	
	
    def render(self, mode='human', close=False):

        pass
	



    def reset(self):
        self.reward = 0 
        self.steps = 0
        self.action_list = [] 

        self.currentData = self.getData().values
		
        observation = np.log10(self.currentData+1)#/np.linalg.norm(self.currentData, ord=1)
        action_obs = np.zeros(observation.shape[1])
        #action_obs[np.array(self.action_list)] = 1
        action_obs = action_obs.reshape(1,observation.shape[1])
        
        observation = np.concatenate((observation, action_obs), axis=1)


        return observation


    def diffEqv2(self,  t, x , nodeIndex=None, nodeVal = 0):
        #nodeIndex = np.array(self.action_list)
        #if nodeIndex is not None:
        #    x[nodeIndex] = nodeVal
        
        k1 = 10**-7
        kr1 = 10**-3
        kc1 =  1 
        
        k2 = 10**-6
        kr2 = 10**-3
        kc2 =  1 
        
        k3 = 3 * 10**-8
        kr3 = 10**-3
        kc3 =  1 
        
        k4 = 10**-7
        kr4 = 10**-3
        kc4 =  1 
        
        k5 = 10**-6
        kr5 = 10**-3
        kc5 =  None
        
        k6 = 2 * 10**-6
        kr6 = 10**-3
        kc6 =  0.01
        
        k7 = 2 * 10**-6
        kr7 = 10**-3
        kc7 =  0.01   
        
        k8 = 10**-4
        kr8 = None
        kc8 =  None
        
        k9 = 5 * 10**-9
        kr9 = 10**-3
        kc9 =  1
        
        k10 = 10**-8
        kr10 = 2 * 10**-4
        kc10 =  1
        
        k11 = 10**-9
        kr11 = 10**-3
        kc11 =  None
        
        k12 = 10**-6
        kr12 = 10**-3
        kc12 =  None
        
        k13 = 3.5 * 10**-6
        kr13 = 10**-3
        kc13 =  None
        
        k14 = 10**-3
        kr14 = 10**-6
        kc14 =  None
        
        k15 = 10**-6
        kr15 = 10**-3
        kc15 =  None
        
        k16 = None
        kr16 = None
        kc16 =  None
        
        k17 = 3.5 * 10**-6
        kr17 = 10**-3
        kc17 =  1
        
        k18 = 10**-9
        kr18 = 10**-3
        kc18 =  1
        
        k19 = 7 * 10**-5
        kr19 = 1.67 * 10**-5
        kc19 =  1.67 * 10**-4
        
        k20 = 10**-8
        kr20 = 1.67 * 10**-3
        kc20 =  1.67 * 10**-2
        
        k21 = 0.01
        kr21 = 0.01
        kc21 =  None
        
        k22 = 10**-7
        kr22 = 10**-3
        kc22 =  1
        
        k23 = 2 * 10**-6
        kr23 = 10**-3
        kc23 =  0.01
        
        s25To41 = 0
        ri_allbut_5 = 5.79 * 10**-6
        r5 = 2.89 * 10**-5
        
        s1 = 0
        s2 = 2.03*10**-2
        
        s3 = 0
        s4 = 5.79*10**-1
        
        s5 = 0
        s6 = 4.63*10**-1
        
        s7 = 0
        s8 = 2.89*10**-1
        
        s9 = 0
        s10 = 2.89
        
        s11 = 0
        s12 = 2.89*10**-1
        
        s13 = 0
        s14 = 2.89
        
        s15 = 5.79 * 10**-1
        s16 = 0
        
        s17 = 0
        s18 = 1.16*10**-1
        
        s19 = 0
        s20 = 1.74*10**-1
        
        s21 = 0
        s22 = 0
        
        s23 = 0
        s24 = 5.79*10**-1
        
        v = 0.07
        
        ep1 = -k1*x[0]*x[1] + kr1*x[25] + s1 - ri_allbut_5*x[0]
        
        ep2 = -k1*x[0]*x[1] + kr1*x[25] - k3*x[4]*x[2] + kr3*x[27] + s2 - ri_allbut_5*x[1] 
        
        ep3 = kc1*x[25] -k2*x[2]*x[3] +kr2*x[26] +kc2*x[26] +kc3*x[27] -k4*x[2]*x[5] +kr4*x[28] +kc4*x[28] -ri_allbut_5*x[2]
        
        ep4 = -k2*x[2]*x[3] +kr2*x[26] -k9*x[10]*x[3] +kr9*x[31] +s4 -ri_allbut_5*x[3]
        
        ep5 = kc2*x[26] -k3*x[4]*x[1] +kr3*x[27] +kc3*x[27] +kc9*x[31] -k19*x[4]*x[19] +kr19*x[36] -k22*x[4]*x[23] +kr22*x[34] +kc22*x[34] -r5*x[4]
        
        ep6 = -k4*x[2]*x[5] +kr4*x[28] +s6 -ri_allbut_5*x[5]
        
        ep7 = kc4*x[28] -k15*x[15]*x[6] +kr15*x[22] -k21*x[6] +kr21*x[21] -ri_allbut_5*x[6]
        
        ep8 = -(1/v**2)* k5*x[21]*x[7] +kr5*x[8] +k8*x[8] +s8 -ri_allbut_5*x[7]
        
        ep9 = (1/v**2)*k5*x[21]*x[7] -kr5*x[8] -(1/v**2)*k6*x[8]*x[9] +kr6*x[29] +kc6*x[29] -(1/v**2)*k7*x[8]*x[11] + kr7*x[30] +kc6*x[30] -k8*x[8]
        
        ep10 = -(1/v**2)*k6*x[8]*x[9] +kr6*x[29] +s10 -ri_allbut_5*x[9]
        
        ep11 = kc6*x[29] -k9*x[10]*x[3] +kr9*x[31] +kc9*x[31] -k11*x[15]*x[10] +kr11*x[16] -k23*x[19]*x[10] +kr23*x[40] -ri_allbut_5*x[10]
        
        ep12 = -(1/v**2)*k7*x[8]*x[11] +kr7*x[30] +s12 -ri_allbut_5*x[11]
        
        ep13 = kc7*x[30] -k13*x[12]*x[19] +kr13*x[20] -k17*x[12]*x[19]*x[15] +kr17*x[33] -ri_allbut_5*x[12]
        
        ep14 = -k10*x[13]*x[15] +kr10*x[32] +kc10*x[32] -k20*x[13]*x[19] +kr20*x[38] +s14 -ri_allbut_5*x[13]
        
        ep15 = kc10*x[32] -k14*x[14] +kr14*x[15] +kc17*x[33] +k18*x[35] +s15 -ri_allbut_5*x[14]
        
        ep16 = -k10*x[13]*x[15] +kr10*x[32] -k11*x[15]*x[10] +kr11*x[16] +k14*x[14] -kr14*x[15] -k15*x[15]*x[6] +kr15*x[22] -k17*x[12]*x[19]*x[15] +kr17*x[33] -k18*x[19]*x[15] +kr18*x[35] -ri_allbut_5*x[15]
        
        ep17 = k11*x[15]*x[10] -kr11*x[16] -ri_allbut_5*x[16]
        
        ep18 = -(1/v**2)*k12*x[17]*x[21] +kr12*x[18] +s18 -ri_allbut_5*x[17]
        
        ep19 = (1/v**2)*k12*x[17]*x[21] -kr12*x[18] -ri_allbut_5*x[18]
        
        ep20 = -k13*x[12]*x[19] +kr13*x[20] -k17*x[12]*x[19]*x[15] +kr17*x[33] -k18*x[15]*x[19] +kr18*x[35]+kc18*x[35] -k19*x[4]*x[19] +kr19*x[36] +kc19*x[36] -k20*x[13]*x[19] +kr20*x[38] +kc20*x[38] -k23*x[19]*x[10]+kr23*x[40] +s20 -ri_allbut_5*x[19]*(1/(x[15]+1))
        
        ep21 = k13*x[12]*x[19] -kr13*x[20] +kc17*x[33] -ri_allbut_5*x[20]
        
        ep22 = -(1/v**2)*k5*x[21]*x[7] +kr5*x[8] -(1/v**2)*k12*x[17]*x[21] +kr12*x[18] +k21*x[6] -kr21*x[21] -ri_allbut_5*x[21] 
        
        ep23 = k15*x[15]*x[6] -kr15*x[22] -ri_allbut_5*x[22]
        
        ep24 = -k22*x[4]*x[23] +kr22*x[34] +s24 -ri_allbut_5*x[23]
        
        ep25 = kc22*x[34] -ri_allbut_5*x[24]
        
        ep26 = k1*x[0]*x[1] -kr1*x[25] -kc1*x[25] -ri_allbut_5*x[25]
        
        ep27 = k2*x[2]*x[3] -kr2*x[26] -kc2*x[26] -ri_allbut_5*x[26]
        
        ep28 = k3*x[4]*x[1] -kr3*x[27] -kc3*x[27] -ri_allbut_5*x[27]
        
        ep29 = k4*x[2]*x[5] -kr4*x[28] -kc4*x[28] -ri_allbut_5*x[28]
    
        ep30 = (1/v**2)*k6*x[8]*x[9] -kr6*x[29] -kc6*x[29] -ri_allbut_5*x[29]
        
        ep31 = (1/v**2)*k7*x[8]*x[11] -kr7*x[30] -kc6*x[30] -ri_allbut_5*x[30]
        
        ep32 = k9*x[10]*x[3] -kr9*x[31] -kc9*x[31] -ri_allbut_5*x[31]
        
        ep33 = k10*x[13]*x[15] -kr10*x[32] -kc10*x[32] -ri_allbut_5*x[32]
        
        ep34 = k17*x[12]*x[19]*x[15] -kr17*x[33] -kc17*x[33] -ri_allbut_5*x[33]
        
        ep35 = k22*x[4]*x[23] -kr22*x[34] -kc22*x[34] -ri_allbut_5*x[34]
        
        ep36 = k18*x[15]*x[19] -kr18*x[35] -kc18*x[35] -ri_allbut_5*x[35]
        
        ep37 = k19*x[4]*x[19] -kr19*x[36] -kc19*x[36] -ri_allbut_5*x[36]
        
        ep38 = kc19*x[36] -ri_allbut_5*x[37] 
        
        ep39 = k20*x[13]*x[19] -kr20*x[38] -kc20*x[38] -ri_allbut_5*x[38] 
        
        ep40 = kc20*x[38] -ri_allbut_5*x[39]
        
        ep41 = k23*x[19]*x[10] -kr23*x[40] -ri_allbut_5*x[40]
        
        
        S = np.array([ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8, ep9, ep10, 
                ep11, ep12, ep13, ep14, ep15, ep16, ep17, ep18, ep19, ep20, 
                ep21, ep22, ep23, ep24, ep25, ep26, ep27, ep28, ep29, ep30, 
                ep31, ep32, ep33, ep34, ep35, ep36, ep37, ep38, ep39, ep40, 
                ep41])
        index = np.where(x <0)[0]
        x[index] = 0
        
        #if nodeIndex is not None:
        #    S[nodeIndex] = nodeVal
    
        return S
    
    def getObsAndReward(self, action):
        self.action_list.append(action)
        
        #print(self.action_list)
        #print(currentData.shape)
        self.currentData[:,action] = 0
        
        t = np.linspace(0, 24*3600, 100*2*3600)
        sol = solve_ivp(self.diffEqv2, (0, 2*3600), self.currentData.reshape(self.currentData.shape[1],), method = 'Radau')
        #print(sol.y.shape)
        self.currentData = sol.y[:,-1].reshape(1,self.currentData.shape[1])
        
        reward = self.getReward(self.currentData.reshape(self.currentData.shape[1]), self.targetState.reshape(self.currentData.shape[1]))
        observation = np.log10(self.currentData+1)
        #observation = self.currentData#/np.linalg.norm(self.currentData, ord=1)
        #print(observation)
        
        action_obs = np.zeros(observation.shape[1])
        action_obs[np.array(self.action_list)] = 1
        action_obs = action_obs.reshape(1,observation.shape[1])
        
        return observation, action_obs, reward
    
    #alloed actions, buy, sell 	
    def step(self, action):
        # action will be a vector of length 41
        
        #action = np.argmax(action)
        print()
        
        print(action)
        print(action)
        print(action)
        
        if action == 1 or action == 35 or action == 32:
            action = 0
        
        observation, action_obs, reward = self.getObsAndReward(action)
        print(observation)
        observation = np.concatenate((observation, action_obs), axis=1)
        #action = action.reshape(3, self.stocks_per_epi)
        #self.currentData = self.currentData *( action+1)
        
        #self.currentData[0,action] = 0

        if np.linalg.norm(self.currentData, ord=1) == 0:
            return observation, -10, True, {'hello':0}
        
        index = np.where(self.targetState.reshape(self.currentData.shape[1]) > 0 )[0]
        if np.all(self.currentData.reshape(self.currentData.shape[1])[index] > self.targetState.reshape(self.currentData.shape[1])[index]):
            return observation, 10, True, {'hello':0}
        
        #observation = observation
        self.steps += 1
        if self.max_step == self.steps:
            return observation, reward, True, {'hello':0}
        

        return observation, reward, False, {'hello':0}
    

    def getReward(self, currentState, targetState):
        index = np.where(targetState > 0 )[0]
        
        v1 = currentState[index]
        v2 = targetState[index]
        
        print(v1,v2)
        print(mean_squared_error(v1,v2))
        print(np.linalg.norm(v2, ord=1))

        reward = -1* np.sqrt(mean_squared_error(v1, v2)) / np.linalg.norm(v2, ord=1)
        
        return np.clip(reward, -1, 1)
        
        # replaced reward with self.reward
        # update holdings
        # update investment

class actionSpace():
	def __init__(self, n):
		self.n = n

	def n():
		return self.n
	def sample(self):
		return np.random.randint(self.n)
		
class observationSpace():
	def __init__(self, n):
		self.n = n

	def n():
		return self.n
