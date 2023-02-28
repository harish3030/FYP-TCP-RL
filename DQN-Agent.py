#!/usr/bin/env python3
# -*- coding: utf-8 -*-




import math
import sys
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error

from ns3gym import ns3env
from tcp_base import TcpTimeBased, TcpEventBased

try:
    w_file = open('run.log', 'w')
except:
    w_file = sys.stdout
parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')
parser.add_argument('--steps',
                    type=int,
                    default=500,
                    help='Number of steps, Default 100')
args = parser.parse_args()

startSim = bool(args.start)
iterationNum = int(args.iterations)
maxSteps = int(args.steps)
print(maxSteps)
port = 5555
simTime = 10000 # seconds
stepTime = 10  # seconds
seed = 12
simArgs = {"--duration": simTime,}
debug = True

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
# simpler:
#env = ns3env.Ns3Env()
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        # we define some parameters and hyperparameters:
        # "lr" : learning rate
        # "gamma": discounted factor
        # "exploration_proba_decay": decay of the exploration probability
        # "batch_size": size of experiences we sample to train the DNN
        self.lr = 0.001
        self.gamma = 0.99
        # self.epsilon = 1.0 # exploration probability at start
        # self.epsilon_min = 0.01 # minimum exploration probability
        # self.epsilon_decay = 0.0005 
        self.exploration_proba = 1.0
        self.exploration_proba_decay = ((2*maxSteps)-1)/(2*maxSteps)
        self.min_epsilon=0.1
        self.batch_size = 32
        
        # We define our memory buffer where we will store our experiences
        # We stores only the 2000 last time steps
        self.memory_buffer= list()
        self.max_memory_buffer = 2000
        
        # We creaate our model having to hidden layers of 24 units (neurones)
        # The first layer has the same size as a state size
        # The last layer has the size of actions space
        self.model = Sequential([
            Dense(units=state_size,input_dim=state_size, activation = 'relu'),
            Dense(units=state_size,activation = 'relu'),
            Dense(units=action_size, activation = 'linear')
        ])
        self.model.compile(loss="mse",
                      optimizer = Adam(lr=self.lr))
        
    # The agent computes the action to perform given a state 
    def compute_action(self, current_state):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action 
        #     with the highest Q-value.
        # exploration_proba= self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * decay_step)
        if np.random.uniform(0,1) < self.exploration_proba:
            action_index=np.random.choice(range(self.n_actions))
            print("\t[*] Random exploration. Selected action: {}".format(action_index), file=w_file)
            return action_index
        q_values = self.model.predict(current_state)[0]
        action_index= np.argmax(q_values)
        print("\t[*] Exploiting gained knowledge. Selected action: {}".format(action_index), file=w_file)
        return action_index

    # when an episode is finished, we update the exploration probability using 
    # espilon greedy algorithm
    def update_exploration_probability(self):
        # self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
        if self.exploration_proba > self.min_epsilon:
           self.exploration_proba*=self.exploration_proba_decay
        #rint(self.exploration_proba)
    
    # At each time step, we store the corresponding experience
    def store_episode(self,current_state, action, reward, next_state, done):
        #We use a dictionnary to store them
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    

    # At the end of each episode, we train our model
    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]
        
        # We iterate over the selected experiences
        for experience in batch_sample:
            # We compute the Q-values of S_t
            q_current_state = self.model.predict(experience["current_state"])
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma*np.max(self.model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            # train the model
            self.model.fit(experience["current_state"], q_current_state, verbose=0)


state_size = ob_space.shape[0] - 4 # ignoring 4 env attributes

action_size = 3
action_mapping = {} # dict faster than list
action_mapping[0] = 0
action_mapping[1] = 600
action_mapping[2] = -150



agent = DQNAgent(state_size, action_size)
total_steps = 0
rewardsum=0

rew_history = []
cWnd_history = []
pred_cWnd_history = []
rtt_history = []
tp_history = []

current_state = env.reset()
current_state = current_state[4:]
cWnd = current_state[1]
init_cWnd = cWnd

current_state = np.array([current_state])

for e in range(iterationNum):
    # We initialize the first state and reshape it to fit 
    #  with the input layer of the DNN
    print(current_state)
    #decay=0
    
    for step in range(maxSteps):

        print("[+] Step: {}".format(step+1), file=w_file)

      
        total_steps = total_steps + 1
        # the agent computes the action to perform
        action = agent.compute_action(current_state)

        
        calc_cWnd = cWnd+action_mapping[action]
           
        new_ssThresh = int(calc_cWnd/2)
        # the envrionment runs the action and returns
        # the next state, a reward and whether the agent is done
        actions = [new_ssThresh, calc_cWnd]
        next_state, reward, done, _ = env.step(actions)
       
        
        rewardsum += reward
        
        next_state = next_state[4:]
        cWnd = next_state[1]
        rtt = next_state[7]
        throughput = next_state[11]
            
        print("\t[#] Next state: ", next_state, file=w_file)
        print("\t[!] Reward: ", reward, file=w_file)
        print("\t[$] Congestion Window: ",cWnd,file=w_file)
        print("\t[!] RTT :",rtt,file=w_file)
        
        rew_history.append(rewardsum)
        rtt_history.append(rtt*1e-6)
        cWnd_history.append(cWnd)
        tp_history.append((throughput*8)/1e6)

        
        next_state = np.array([next_state])
        # We sotre each experience in the memory buffer
        agent.store_episode(current_state, action, reward, next_state, done)
        
        # if the episode is ended, we leave the loop after
        # updating the exploration probability
        if done:
            print("[X] Stopping: step: {}, reward sum: {}"
                        .format(step+1, rewardsum),
                        file=w_file)
            
            break
        agent.update_exploration_probability()    
        current_state = next_state
    
    print("\n[O] Iteration over.", file=w_file)
    print("[-] Final reward sum: ", rewardsum, file=w_file)
    print()
    # if the have at least batch_size experiences in the memory buffer
    # than we tain our model
    #batch_size=32
    if total_steps >= agent.batch_size:
        agent.train()
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(2, 2, figsize=(4,2))
plt.tight_layout(pad=0.3)

#ew_history

rew_history.pop()
rtt_history.pop()
cWnd_history.pop()
tp_history.pop()

ax[0, 0].plot(range(len(cWnd_history)), cWnd_history, marker="", linestyle="-")
ax[0, 0].set_title('Congestion windows')
ax[0, 0].set_xlabel('Steps')
ax[0, 0].set_ylabel('CWND (segments)')

ax[0, 1].plot(range(len(tp_history)), tp_history, marker="", linestyle="-")
ax[0, 1].set_title('Throughput over time')
ax[0, 1].set_xlabel('Steps')
ax[0, 1].set_ylabel('Throughput (Mbits/s)')

ax[1, 0].plot(range(len(rtt_history)), rtt_history, marker="", linestyle="-")
ax[1, 0].set_title('RTT over time')
ax[1, 0].set_xlabel('Steps')
ax[1, 0].set_ylabel('RTT (seconds)')

ax[1, 1].plot(range(len(rew_history)), rew_history, marker="", linestyle="-")
ax[1, 1].set_title('Reward sum plot')
ax[1, 1].set_xlabel('Steps')
ax[1, 1].set_ylabel('Accumulated reward')

#plt.xlim([0, 1000])
plt.savefig('plots.png')
plt.show()        