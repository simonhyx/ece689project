from __future__ import division
import argparse

#from PIL import Image
import numpy as np
#import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K


from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import glob
import pandas as pd
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Conv1D
from keras.layers import GlobalAveragePooling2D
from keras.layers import LSTM,Dense, TimeDistributed, Reshape, Lambda
from keras.models import Model
from keras.activations import softmax
from keras.backend import argmax
import keras

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *
from rl.agents import DDPGAgent
from rl.agents import DQNAgent
#from ddpg import DDPGAgent
import copy

#from Env_fixpoint_dqn import controlEnv
from dataGeneratorEdge_redo_v2 import dataGenerator


from Env_dqn_alter import controlEnv



'''
path = './simulatedData/*.npy'
files = glob.glob(path)


steadyStateData = []
for file in files:
    data = np.load(file)
    steadyStateData.append(data[-1,:])


steadyStateData = np.array(steadyStateData)
np.save("data", steadyStateData)

steadyStateData = np.load("data.npy")
steadyStateData2 = np.load("data.npy")
'''

steadyStateData = np.load("dataNormal.npy")
steadyStateData2 = np.load("dataAlter.npy")
generator = dataGenerator()
colNames = generator.variableList()
connection = generator.generateGraph()

df = pd.DataFrame(steadyStateData, columns=colNames)
dfAlter = pd.DataFrame(steadyStateData2, columns=colNames)

targetState = copy.deepcopy(df.loc[0:0])
for element in targetState.columns:
    targetState[element] = -1
targetState['C3*'] = df['C3*'].mean() + 2*df['C3*'].std()
env = controlEnv(df, colNames, targetState.values.reshape(41,))




#env = gym.make(args.env_name)
np.random.seed(123)
#env.seed(123)
nb_actions = env.action_space.n



class AtariProcessor(Processor):
    def process_observation(self, observation):
        #print(observation)
        #print(observation[1].shape)
        return observation
    def process_state_batch(self, batch):

        all_obs = []
        all_states = []
        for element in batch:
            all_obs.append(element[0][0])

        all_obs = np.array(all_obs)


        return all_obs
        #return [batch[0,0,0].reshape(1,batch[0,0,0].shape[0], batch[0,0,0].shape[1]), batch[0,0,1].reshape(1,batch[0,0,1].shape[0])]

    def process_reward(self, reward):
        return reward


processor = AtariProcessor()

def actor_net(in_shape, n_classes = 2):
    img = Input(in_shape)

    net = Dense(32, activation = 'relu')(img)
#	net = LSTM(8,  return_sequences = True)(net)
    net = Dense(32, activation = 'relu')(net)
    
    out = Dense(n_classes, activation = 'sigmoid')(net)
#
    model = Model(inputs=img,outputs=out)
    return model


model = actor_net((env.obs_space,),nb_actions)
#print(actor.summary())




# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=10)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

dqn.fit(env, nb_steps=1750000, log_interval=10000, nb_max_episode_steps=5)



