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
parser.add_argument('--bandwidth',
                    type=int,
                    default=1e7,
                    help='Number of steps, Default 100')
parser.add_argument('--leafs',
                    type=int,
                    default=5,
                    help='Number of steps, Default 100')
parser.add_argument('--attacker',
                    type=int,
                    default=2,
                    help='Number of steps, Default 100')
args = parser.parse_args()

startSim = bool(args.start)
iterationNum = int(args.iterations)
maxSteps = int(args.steps)
bandwidth=int(args.bandwidth)
leafs=int(args.leafs)
attacker=int(args.attacker)
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
        self.exploration_proba_decay = ((iterationNum*maxSteps)-1)/(iterationNum*maxSteps)
        # self.exploration_proba_decay = 0.1
        #self.exploration_proba_decay = 1/(iterationNum*maxSteps)
        # self.exploration_proba_decay = 0.005
        self.min_epsilon=0.1
        self.batch_size = 3
        
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
    def compute_action(self, current_state,node_id):
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
        if node_id==attacker:
            action_index=np.argmin(q_values)
        print("\t[*] Exploiting gained knowledge. Selected action: {}".format(action_index), 
              file=w_file)
        return action_index
    
    def predict_action(self,current_state):
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
        # self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
        
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
    def train(self,node_id):
        # We shuffle the memory buffer and select a batch size of experiences
        #np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[-1:-1-self.batch_size:-1]
        
        # We iterate over the selected experiences
        for experience in batch_sample:
            # We compute the Q-values of S_t
            q_current_state = self.model.predict(experience["current_state"])
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"] and node_id!=attacker:
                q_target = q_target + self.gamma*np.max(self.model.predict(
                                                     experience["next_state"])[0])
            elif not experience["done"] and node_id==attacker:
                q_target = q_target + self.gamma*np.min(self.model.predict(
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
# action_mapping[3]=3


agent = DQNAgent(state_size, action_size)
total_steps = 0
rewardsum=0
N=40
rew_history = [[] for i in range(leafs)]
cWnd_history = [[] for i in range(leafs)]
rtt_history = [[] for i in range(leafs)]
tp_history = [[] for i in range(leafs)]
tp=[]
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
col=["red","green","blue","pink","grey","brown"]
def plot_individual_graphs(history,tit,label,episode):
   
    
    plt.title(tit)
    plt.xlabel('Steps')
    plt.ylabel(label)
    leg=[]
    # for i in range(1,leafs):
    #     while len(history[i])!=len(history[0]):
    #         history[i].append(history[i][-1])
    for i in range(leafs):
       plt.plot(range(len(history[i])),history[i],marker="",linestyle="-",color=col[i])
       leg.append(str(i+1))
    plt.legend(leg)   
    
    output_dir = f"./Attack_Plots/Episode-{episode}"
    mkdir_p(output_dir)
    file_name=f'{tit}.png'
    plt.savefig(f"{output_dir}/{file_name}",bbox_inches='tight')
    plt.show()
def plot_graphs(history,tit,label,episode):
    #print(history)
    episode=episode+1
    
    plt.plot(range(len(history)),history,marker="",linestyle="-",color="red")
    plt.title(tit)
    plt.xlabel('Episodes')
    plt.ylabel(label)
    output_dir = f"./Attack_Plots/Episode-{episode}"
    mkdir_p(output_dir)
    file_name=f'{tit}.png'
    plt.savefig(f"{output_dir}/{file_name}",bbox_inches='tight')
    plt.show() 
def utility(throughput_c, delay_c):
    
    r = 0
    if throughput_c > 0:
      r += math.log(throughput_c/bandwidth)
    if delay_c > 0:
      # r += -0.2 * math.log(delay_c);
      r += -0.01 * math.log(delay_c)
    
    return r

def find_reward(u2, u1):
    diff = u2 - u1
    if diff >= 1:
      return 10
    elif 0<= diff < 1:
      return 5
    elif -1 <= diff < 0:
      return -5
    else:
      return -10

def perturb(current_state,cWnd,node_id):
    old_action=agent.compute_action(current_state,node_id)
    old_cWnd = cWnd+action_mapping[old_action]
    adversarial_noise=0
    action=old_action
    sampling_count=20
    epsilon=0.2
    #Sampling noise
    for i in range(sampling_count):
        random_noise = np.random.uniform(-1,1,1)-0.5
        new_state=current_state+epsilon*random_noise
        action=agent.compute_action(new_state,node_id)
        new_cWnd = cWnd+action_mapping[action]
        if new_cWnd<old_cWnd:
            old_cWnd=new_cWnd
            adversarial_noise=random_noise
    return [adversarial_noise,action,old_cWnd]


for e in range(iterationNum):
    current_state = env.reset()
    node_id=current_state[3]
    current_state = current_state[4:]
    cWnd = current_state[1]
    init_cWnd = cWnd
    current_state = np.array([current_state])
    running_tp=0
    running_rew=0
    running_rtt=0
    old_U=0
    for step in range(maxSteps):

        print("[+] Iteration:{} Step: {}".format(e+1,step+1), file=w_file)
        
        total_steps = total_steps + 1
        if node_id==attacker:
            adversarial_noise,action,calc_cWnd=perturb(current_state,cWnd,node_id)
            # if adversarial_noise>0:
            #     calc_cWnd=cWnd+1000
            # elif adversarial_noise<0:
            #     calc_cWnd=cWnd-1000
            # else:
            #     calc_cWnd=cWnd    

        # the agent computes the action to perform
        else:
            action = agent.compute_action(current_state,node_id)
            calc_cWnd = cWnd+action_mapping[action]
        
    
        if calc_cWnd<0:
            calc_cWnd=cWnd
        # set new ssthresh   
        new_ssThresh = int(calc_cWnd/2)
        # the envrionment runs the action and returns
        # the next state, a reward and whether the agent is done
        actions = [new_ssThresh, calc_cWnd]
        next_state, reward, done, _ = env.step(actions)
        
        new_U=utility(next_state[15],(next_state[11]-next_state[12]))
        reward=find_reward(new_U,old_U)
        old_U=new_U
        rewardsum += reward
        node_id=next_state[3]
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
        cWnd_history[node_id-2].append(cWnd)
        rew_history[node_id-2].append(rewardsum)
        rtt_history[node_id-2].append(rtt*1e-6)
        if throughput==0:
            tp.append((step,e))
        
        tp_history[node_id-2].append(throughput*8)

        running_tp=running_tp+(throughput*8)
        running_rew=running_rew+rewardsum
        running_rtt=running_rtt+(rtt*1e-6)
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
           agent.train(node_id)
           total_steps=0

        #update epsilon value
        agent.update_exploration_probability()  
        # agent.save_weights()
        
        # agent.update_exploration_probability() 
        current_state = next_state
    
    # if total_steps >= agent.batch_size:
    #        agent.train()
      
    agent.save_model() 
    print("\n[O] Iteration over.", file=w_file)
    print("[-] Final reward sum: ", rewardsum, file=w_file)
    print()
    global_tp_history.append(running_tp//(len(global_tp_history)+1))
    global_rew_history.append(running_rew/(len(global_rew_history)+1))
    global_rtt_history.append(running_rtt/(len(global_rtt_history)+1))
    mpl.rcdefaults()
    mpl.rcParams.update({'font.size': 16})
    plot_individual_graphs(cWnd_history,'Congestion Windows','CWND',e)
    plot_individual_graphs(tp_history,'Throughput over time','Throughput (Bits/s)',e)
    plot_individual_graphs(rtt_history,'RTT over time','RTT (seconds)',e)
    plot_individual_graphs(rew_history,'Reward sum plot','Accumulated reward',e)
    plot_graphs(global_tp_history,'Avg Throughput over time','Throughput (Bits/s)',e)
    plot_graphs(global_rtt_history,'Avg RTT over time','RTT (seconds)',e)
    plot_graphs(global_rew_history,'Avg Reward sum plot','Accumulated reward',e)
    rew_history = [[] for i in range(leafs)]
    cWnd_history = [[] for i in range(leafs)]
    rtt_history = [[] for i in range(leafs)]
    tp_history = [[] for i in range(leafs)]
    
#     # if the have at least batch_size experiences in the memory buffer
#     # than we tain our model
    

