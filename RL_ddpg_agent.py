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

from Env_ddpg_multipleStates import controlEnv
from dataGeneratorEdge_redo_v2 import dataGenerator

'''
path = './simulatedData/*.npy'
files = glob.glob(path)


steadyStateData = []
for file in files:
    data = np.load(file)
    steadyStateData.append(data[-1,:])


steadyStateData = np.array(steadyStateData)
np.save("data", steadyStateData)
'''


steadyStateData = np.load("dataNormal.npy")
steadyStateData2 = np.load("dataAlter.npy")
generator = dataGenerator()
colNames = generator.variableList()
connection = generator.generateGraph()

dfNormal = pd.DataFrame(steadyStateData, columns=colNames)
dfAlter = pd.DataFrame(steadyStateData2, columns=colNames)
'''
steadyStateData = np.load("data.npy")

generator = dataGenerator()
colNames = generator.variableList()
connection = generator.generateGraph()

df = pd.DataFrame(steadyStateData, columns=colNames)
'''

targetState = copy.deepcopy(dfNormal.loc[0:0])
for element in targetState.columns:
    targetState[element] = -1
targetState['C3*'] = dfNormal['C3*'].max() *2
dfAlter['C3*'].max() *2

allowed_actions = ['pC8', 'pC3', 'BAX', 'XIAP' , 'TRAIL']

env = controlEnv(dfNormal, dfAlter, allowed_actions, targetState.values.reshape(41,))



#env = gym.make(args.env_name)
np.random.seed(123)
#env.seed(123)
nb_actions = env.action_space.n
'''
# Next, we build our model. We use the same model that was described by Mnih et al. (2015).


def Unit(x, nb_actions, pool=False):
    res = x
    out = Dense(nb_actions, activation = "softmax")(res)
    return out

def rnn(in_shape, n_classes = 2):
	img = Input(in_shape)
	#net = K.permute_dimensions(img, (1,0,2))
	net = LSTM(64)(img)
	net = Dense(n_classes, activation='linear')(net)
	#net = Reshape((n_classes,3))(net)
	#out = Lambda(lambda x: softmax(x, axis=2))(net)
	#out = Lambda(lambda x: argmax(x, axis=-1))(out)
	model = Model(img,outputs=net)
	return model



model = rnn((30,nb_actions),nb_actions)
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

'''

class AtariProcessor(Processor):
    def process_observation(self, observation):
        #print(observation)
        #print(observation[1].shape)
        return observation
    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        #print(batch)
        #processed_batch = batch.astype('float32') 
        #print(batch.shape)
        #print(batch)
        #print(batch[0,0,1])
        #print(batch[0,0,1].shape)
        #print(batch[0,0,1])
        all_obs = []
        all_states = []
        for element in batch:
            all_obs.append(element[0][0])

        all_obs = np.array(all_obs)

        #print(all_obs.shape)
        #print(all_states.shape)
        #print("len of batch = 0"+ str(len(batch)))
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
    
    out = Dense(n_classes, activation = 'linear')(net)
#
    model = Model(inputs=img,outputs=out)
    return model


actor = actor_net((env.obs_space,),nb_actions)
print(actor.summary())

action_space = (nb_actions,)
observation_space = (env.obs_space,)


action_input = Input(action_space, name='action_input')

observation_input = Input(observation_space, name = "obs_input")


#flattened_observation = Flatten()(net)
x = Concatenate()([action_input, observation_input])
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)

"""
def critics(action_space, observation_space, selfState_space):
    action_input = Input(action_space, name='action_input')
    observation_input = Input(observation_space, name = "obs_input")
    flattened_observation = Flatten()(observation_input)
    selfState_input = Input(selfState_space, name = "selfState_input")
    x = Concatenate()([action_input, flattened_observation, selfState_input])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input, selfState_input], outputs=x)
    return critic
critic = critics((nb_actions,), (30,int(nb_actions/3)), (int(nb_actions/3+2),) )
"""
print(critic.summary())
"""
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input((30,int(nb_actions/3)))
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())
"""
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,processor=processor,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)



agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mse'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=90000, visualize=True, verbose=1, nb_max_episode_steps=50)



agent.save_weights('ddpg_agent_weights.h5f', overwrite=True)
# After training is done, we save the final weights.





