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
                    default=1000,
                    help='Number of steps, Default 100')
args = parser.parse_args()

startSim = bool(args.start)
iterationNum = int(args.iterations)
maxSteps = int(args.steps)
print(maxSteps)
port = 5556
simTime = 10000 # seconds
stepTime = 0.5 # seconds
seed = 12
simArgs = {"--duration": simTime,}
debug = True

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, 
                    simArgs=simArgs, debug=debug)
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        # "lr" : learning rate
        # "gamma": discounted factor
        # "exploration_proba_decay": decay of the exploration probability
        # "batch_size": size of experiences we sample to train the DNN
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = ((2*iterationNum*maxSteps)-1)/(2*iterationNum*maxSteps)
        #self.exploration_proba_decay = 1/(iterationNum*maxSteps)
        #self.exploration_proba_decay = 0.005
        self.min_epsilon=0.1
        self.batch_size = 32
        
        # We define our memory buffer where we will store our experiences
        # We stores only the 2000 last time steps
        self.memory_buffer= list()
        self.max_memory_buffer = 1200
        
        # We creaate our model having to hidden layers of 24 units (neurones)
        # The first layer has the same size as a state size
        # The last layer has the size of actions space
        
        self.model = Sequential([
            Dense(units=state_size,input_dim=state_size, activation = 'elu'),
            Dense(units=state_size,activation = 'elu'),
            Dense(units=action_size, activation = 'linear')
        ])
        self.model.compile(loss="mse",
                      optimizer = Adam(lr=self.lr),
                      metrics=['accuracy'])
        
    # The agent computes the action to perform given a state 
    def compute_action(self, current_state):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action 
        #     with the highest Q-value.
        if np.random.uniform(0,1) < self.exploration_proba:
            action_index=np.random.choice(range(self.n_actions))
            print("\t[*] Random exploration. Selected action: {}".format(action_index), 
                  file=w_file)
            return action_index
        q_values = self.model.predict(current_state)[0]
        action_index= np.argmax(q_values)
        print("\t[*] Exploiting gained knowledge. Selected action: {}".format(action_index), 
              file=w_file)
        return action_index
    
    #Save Model
    def save_model(self):
        self.model.save("DQN_model")   
    
    def load_model(self):
        self.model=tf.keras.models.load_model("DQN_model")
    
    #Save weights
    def save_weights(self):
        self.model.save_weights('./checkpoints/my_checkpoint')  
    
    def load_weights(self):
        self.model.load_weights('./checkpoints/my_checkpoint')

    # we update the exploration probability using 
    # espilon greedy algorithm
    def update_exploration_probability(self):
         if self.exploration_proba > self.min_epsilon:
            self.exploration_proba*=self.exploration_proba_decay
        
    def find_exploration_proba(self):
        return self.exploration_proba
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
        #np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]
        
        # We iterate over the selected experiences
        for experience in batch_sample:
            # We compute the Q-values of S_t
            q_current_state = self.model.predict(experience["current_state"])
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma*np.max(self.model.predict(
                                                     experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            # train the model
            self.model.fit(experience["current_state"], q_current_state, verbose=0)


state_size = ob_space.shape[0] - 4 # ignoring 4 env attributes

action_size = 3
action_mapping = {} # dict faster than list
action_mapping[0] = 0
action_mapping[1]=600
action_mapping[2]=-150
# action_mapping[0]=0
# action_mapping[1]=1
# action_mapping[2]=-1
# action_mapping[3]=2
# action_mapping[4]=3

agent = DQNAgent(state_size, action_size)
# agent.load_model()
total_steps = 0
rewardsum=0

rew_history = []
cWnd_history = []
pred_cWnd_history = []
rtt_history = []
tp_history = []
tp=[]
global_cWnd_history=[]
global_rtt_history=[]
global_rew_history=[]
global_tp_history=[]
# current_state = env.reset()
# # print(current_state)
# current_state = current_state[4:]
# cWnd = current_state[1]
# init_cWnd = cWnd

# current_state = np.array([current_state])
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
def plot_graphs(history,tit,label,episode):

    plt.plot(range(len(history)),history,marker="",linestyle="-")
    plt.title(tit)
    plt.xlabel('Steps')
    plt.ylabel(label)
    output_dir = f"./Plots/Episode-{episode}"
    mkdir_p(output_dir)
    file_name=f'{tit}.png'
    plt.savefig(f"{output_dir}/{file_name}")
    plt.show()

# agent.load_weights()
for e in range(iterationNum):
    current_state = env.reset()
    current_state = current_state[4:]
    cWnd = current_state[1]
    init_cWnd = cWnd
    current_state = np.array([current_state])
    for step in range(maxSteps):

        print("[+] Iteration:{} Step: {}".format(e+1,step+1), file=w_file)
        
        total_steps = total_steps + 1
        # the agent computes the action to perform
        action = agent.compute_action(current_state)
        # change CWND
        calc_cWnd = cWnd+action_mapping[action]
    
        if calc_cWnd<0:
            calc_cWnd=cWnd
        # set new ssthresh   
        new_ssThresh = int(calc_cWnd/2)
        # the envrionment runs the action and returns
        # the next state, a reward and whether the agent is done
        actions = [new_ssThresh, calc_cWnd]
        next_state, reward, done, _ = env.step(actions)
       
        rewardsum += reward
        #print("Next state: ",next_state)
        next_state = next_state[4:]
        # print(next_state)
        cWnd = next_state[1]
        rtt = next_state[7]
        throughput = next_state[11]
        
        print("\t[#] Next state: ", next_state, file=w_file)
        print("\t[!] Reward: ", reward, file=w_file)
        print("\t[$] Congestion Window: ",cWnd,file=w_file)
        print("\t[!] RTT :",rtt,file=w_file)
        print("\t[!] Throughput: ",throughput,file=w_file)
        print("\t[!]Epsilon: ",agent.find_exploration_proba(),file=w_file)
        rew_history.append(rewardsum)
        rtt_history.append(rtt*1e-6)
        cWnd_history.append(cWnd)
        if throughput==0:
            tp.append((step,e))
        tp_history.append(throughput*8)
        global_tp_history.append((throughput*8))
        global_rew_history.append(rewardsum)
        global_rtt_history.append(rtt*1e-6)
        global_cWnd_history.append(cWnd)
        
        next_state = np.array([next_state])
        # We store each experience in the memory buffer
        agent.store_episode(current_state, action, reward, next_state, done)
        
        # if the episode is ended, we leave the loop
        if done:
            print("[X] Stopping: step: {}, reward sum: {}"
                        .format(step+1, rewardsum),
                        file=w_file)
            break
        if total_steps >= agent.batch_size:
           agent.train()
        #update epsilon value
        agent.update_exploration_probability()  
        # agent.save_weights()
        agent.save_model()
        current_state = next_state
    
       
    
    print("\n[O] Iteration over.", file=w_file)
    print("[-] Final reward sum: ", rewardsum, file=w_file)
    print()
    mpl.rcdefaults()
    mpl.rcParams.update({'font.size': 16})
    plot_graphs(cWnd_history,'Congestion Windows','CWND (segments)',e)
    plot_graphs(tp_history,'Throughput over time','Throughput (Mbits/s)',e)
    plot_graphs(rtt_history,'RTT over time','RTT (seconds)',e)
    plot_graphs(rew_history,'Reward sum plot','Accumulated reward',e)
    cWnd_history=[]
    tp_history=[]
    rtt_history=[]
    rew_history=[]
    
    # if the have at least batch_size experiences in the memory buffer
    # than we tain our model
    


print(tp)       
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})
plot_graphs(global_cWnd_history,'Congestion Windows','CWND (segments)',iterationNum)
plot_graphs(global_tp_history,'Throughput over time','Throughput (Mbits/s)',iterationNum)
plot_graphs(global_rtt_history,'RTT over time','RTT (seconds)',iterationNum)
plot_graphs(global_rew_history,'Reward sum plot','Accumulated reward',iterationNum)