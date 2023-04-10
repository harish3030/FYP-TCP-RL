#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import math
import sys
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from agent import Agent

from utils import plot_learning_curve
import tensorflow as tf
from datetime import datetime
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

state_size = ob_space.shape[0] - 4 # ignoring 4 env attributes

action_size = 3
action_mapping = {} # dict faster than list
action_mapping[0] = 0
action_mapping[1]=600
action_mapping[2]=-150

N = 20
batch_size = 32
n_epochs = 10
alpha = 0.0003
agent = Agent(n_actions=3, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)

    
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
total_steps = 0
rewardsum=0
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
        action, prob, val = agent.choose_action(current_state)
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
        # print("\t[!]Epsilon: ",agent.find_exploration_proba(),file=w_file)
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
        #agent.store_episode(current_state, action, reward, next_state, done)
        agent.store_transition(current_state, action,
                                   prob, val, reward, done)
        
        # if the episode is ended, we leave the loop
        if done:
            print("[X] Stopping: step: {}, reward sum: {}"
                        .format(step+1, rewardsum),
                        file=w_file)
            break
        if total_steps%N==0:
           agent.learn()
        #update epsilon value
        #agent.update_exploration_probability()  
        # agent.save_weights()
        agent.save_models()
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
    
    # N = 20
    # batch_size = 5
    # n_epochs = 4
    # alpha = 0.0003
    # agent = Agent(n_actions=3, batch_size=batch_size,
    #               alpha=alpha, n_epochs=n_epochs,
    #               input_dims=env.observation_space.shape)
    # n_games = 300

    # best_score = env.reward_range[0]
    # score_history = []

    # learn_iters = 0
    # avg_score = 0
    # n_steps = 0

    # for i in range(n_games):
    #     observation = env.reset()
    #     done = False
    #     score = 0
    #     while not done:
    #         action, prob, val = agent.choose_action(np.expand_dims(observation, axis=0))
    #         observation_, reward, done, info = env.step(action)
    #         n_steps += 1
    #         score += reward
    #         agent.store_transition(observation, action,
    #                                prob, val, reward, done)
    #         if n_steps % N == 0:
    #             agent.learn()
    #             learn_iters += 1
    #         observation = observation_
    #     score_history.append(score)
    #     avg_score = np.mean(score_history[-100:])
        
    #     if avg_score > best_score:
    #         best_score = avg_score
    #         agent.save_models()
    #     tf.summary.scalar('reward summary', data=avg_score, step=i)
    #     print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
    #           'time_steps', n_steps, 'learning_steps', learn_iters)
    #     env.render()
    # filename = 'PPO_trading_view.png'
    # x = [i+1 for i in range(len(score_history))]
    # plot_learning_curve(x, score_history, figure_file)