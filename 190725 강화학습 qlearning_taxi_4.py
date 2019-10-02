# -*- coding: utf-8 -*-
"""
Example Design: Self-Driving Cab
Solving the Cab Problem using Q Learning
https://gym.openai.com/envs/Taxi-v2/
"""
"""
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow). 
    When the episode starts, the taxi starts off at a random square and the passenger is at a random location. 
    The taxi drive to the passenger's location, pick up the passenger, 
    drive to the passenger's destination (another one of the four specified locations), 
    and then drop off the passenger. Once the passenger is dropped off, the episode ends.

    Observations: 
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations 
    of the passenger (including the case when the passenger is the taxi), and 4 destination locations. 
    
    Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. 
    There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, B and Y): locations for passengers and destinations

    actions:
    - 0: south
    - 1: north
    - 2: east
    - 3: west
    - 4: pickup
    - 5: dropoff

    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
"""

#import numpy as np
import random

import gym
env = gym.make('Taxi-v2')

#​env.render()

q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s,a)] = 0.0
        
def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
    qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
    q[(prev_state,action)] += alpha * (reward + gamma * qa - q[(prev_state,action)])
    
def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: q[(state,x)])    
    
alpha = 0.4
gamma = 0.999
epsilon = 0.017

for i in range(8000):
    r = 0
    
    prev_state = env.reset()
    
    while True:
        
        env.render()
        
        # In each state, we select the action by epsilon-greedy policy
        action = epsilon_greedy_policy(prev_state, epsilon)
        
        # then we perform the action and move to the next state, and receive the reward
        nextstate, reward, done, _ = env.step(action)
        
        # Next we update the Q value using our update_q_table function
        # which updates the Q value by Q learning update rule
        
        update_q_table(prev_state, action, reward, nextstate, alpha, gamma)
        
        # Finally we update the previous state as next state
        prev_state = nextstate

        # Store all the rewards obtained
        r += reward

        #we will break the loop, if we are at the terminal state of the episode
        if done:
            break

    print("total reward: ", r)

env.close()