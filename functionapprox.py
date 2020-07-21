import numpy as np

from collections import defaultdict
from functools import partial
import gym
import tiles3


def argmax_random(a):
    a = np.array(a)
    return np.random.choice(np.flatnonzero(a == a.max()))

class MountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = tiles3.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
    
    def get_tiles(self, state):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.
        
        Arguments:
        position -- float, the position of the agent between -1.2 and 0.5
        velocity -- float, the velocity of the agent between -0.07 and 0.07
        returns:
        tiles - np.array, active tiles
        """
        position, velocity = state
        POSITION_MIN = -1.2
        POSITION_MAX = 0.5
        VELOCITY_MIN = -0.07
        VELOCITY_MAX = 0.07
        
        # Scale position and velocity by multiplying the inputs of each by their scale
        
        
        position_scale = self.num_tiles / (POSITION_MAX - POSITION_MIN)
        velocity_scale = self.num_tiles / (VELOCITY_MAX - VELOCITY_MIN)
        
        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here
        tiles = tiles3.tiles(self.iht, self.num_tilings, [position * position_scale, 
                                                      velocity * velocity_scale])
        
        return np.array(tiles)



num_tilings = 8
num_tiles = 8
iht_size = 4096
num_actions = 3

class SarsaAgent:
    def __init__(self, action_count, discount, step_size):
        ''' Creates an Agent that learns an Epsilon Greedy Policy using the SARSA algorithm'''
        self.action_count = action_count

        self.tc = MountainCarTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        self.w = np.ones((num_actions, iht_size)) * 0.0
        self.discount = discount
        self.step_size = step_size

    def select_action(self, state, epsilon=0.2):
        state = self.tc.get_tiles(state)

        action_values = [sum(q[state]) for q in self.w]
        if np.random.random() < epsilon:
            return np.random.choice(num_actions)
        else:
            return argmax_random(action_values)



    def update_policy(self, state, action, reward, next_state, next_action):
        state = self.tc.get_tiles(state)
        next_state = self.tc.get_tiles(next_state)

        # Qlearning
        # next_action_value = max([sum(q[next_tiles]) for q in w])
        next_action_value = sum(self.w[next_action, next_state])
        delta = reward + self.discount*next_action_value - sum(self.w[action, state])

        self.w[action, state] += self.step_size*delta
    
    def update_policy_on_end(self, state, action, reward):
        state = self.tc.get_tiles(state)

        delta = reward - sum(self.w[action, state]) 
        self.w[action, state] += self.step_size*delta



class QLearningAgent:
    def __init__(self, action_count, discount, step_size):
        ''' Creates an Agent that learns an Epsilon Greedy Policy using the SARSA algorithm'''
        self.action_count = action_count

        self.tc = MountainCarTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        self.w = np.ones((num_actions, iht_size)) * 0.0
        self.discount = discount
        self.step_size = step_size

    def select_action(self, state, epsilon=0.2):
        state = self.tc.get_tiles(state)

        action_values = [sum(q[state]) for q in self.w]
        if np.random.random() < epsilon:
            return np.random.choice(num_actions)
        else:
            return argmax_random(action_values)



    def update_policy(self, state, action, reward, next_state):
        state = self.tc.get_tiles(state)
        next_state = self.tc.get_tiles(next_state)

        # Qlearning
        next_action_value = max([sum(q[next_state]) for q in self.w])
        # next_action_value = sum(self.w[next_action, next_state])
        delta = reward + self.discount*next_action_value - sum(self.w[action, state])

        self.w[action, state] += self.step_size*delta
    
    def update_policy_on_end(self, state, action, reward):
        state = self.tc.get_tiles(state)

        delta = reward - sum(self.w[action, state]) 
        self.w[action, state] += self.step_size*delta


def show_episode(env, agent):
    action_count = env.action_space.n
    state =  env.reset() # start state

    done = False
    while not done:
        env.render()

        action = agent.select_action(state, epsilon=0)
        state, reward, done, info = env.step(action) 

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    action_count = env.action_space.n

    step_size = 0.5 / num_tilings
    discount = 1.0
    epsilon = 0.2


    # agent = SarsaAgent(action_count, discount, step_size)
    agent = QLearningAgent(action_count, discount, step_size)

    for episode in range(100):
        state = env.reset()
        action = agent.select_action(state, epsilon)

        total_reward = 0
        done = False
        while not done:
            next_state, reward, done, info = env.step(action) 

            total_reward += reward

            if done:
                agent.update_policy_on_end(state, action, reward)
                break
            else:
                next_action = agent.select_action(next_state, epsilon)
                
                # agent.update_policy(state, action, reward, next_state, next_action)
                agent.update_policy(state, action, reward, next_state)


            state = next_state
            action = next_action
        
        if episode % 25 == 0:
            print (total_reward)

            show_episode(env, agent)

    env.close()