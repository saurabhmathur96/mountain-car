import numpy as np

from collections import defaultdict
from functools import partial
import gym

def argmax_random(a):
    a = np.array(a)
    return np.random.choice(np.flatnonzero(a == a.max()))

class MountainCarDiscretizer:
    def __init__(self, bin_sizes):
        self.bin_sizes = bin_sizes
        self.ranges = [[-1.2, 0.6], [-0.07 , 0.07]]

    def discretize(self, observations):
        discrete = []
        for observation, r, bin_size in zip(observations, self.ranges, self.bin_sizes):
            bins = np.linspace(*r, num=bin_size)
            discrete.append(np.digitize(observation, bins))

        return tuple(discrete)


class SarsaAgent:
    def __init__(self, action_count, discount, step_size):
        ''' Creates an Agent that learns an Epsilon Greedy Policy using the SARSA algorithm'''
        self.action_count = action_count
        bins = [7, 7]
        self.discretizer = MountainCarDiscretizer(bin_sizes=bins)
        self.Q = defaultdict(partial(np.zeros, (action_count,)))
        self.discount = discount
        self.step_size = step_size

    def select_action(self, state, epsilon=0.2):
        state = self.discretizer.discretize(state)

        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_count)
        else:
            return argmax_random(self.Q[state])

    def update_policy(self, state, action, reward, next_state, next_action):
        state = self.discretizer.discretize(state)
        next_state = self.discretizer.discretize(next_state)

        target = reward + self.discount*self.Q[next_state][next_action]
        delta = target - self.Q[state][action]
        self.Q[state][action] += self.step_size*delta
    
    def update_policy_on_end(self, state, action, reward):
        state = self.discretizer.discretize(state)
        delta = - self.Q[state][action]
        self.Q[state][action] += self.step_size*delta 


class QLearningAgent:
    def __init__(self, action_count, discount, step_size):
        ''' Creates an Agent that learns an Epsilon Greedy Policy using the SARSA algorithm'''
        self.action_count = action_count
        bins = [7, 7]
        self.discretizer = MountainCarDiscretizer(bin_sizes=bins)
        self.Q = defaultdict(partial(np.zeros, (action_count,)))
        self.discount = discount
        self.step_size = step_size

    def select_action(self, state, epsilon=0.2):
        state = self.discretizer.discretize(state)

        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_count)
        else:
            return argmax_random(self.Q[state])

    def update_policy(self, state, action, reward, next_state):
        state = self.discretizer.discretize(state)
        next_state = self.discretizer.discretize(next_state)

        target = reward + self.discount*self.Q[next_state].max()
        delta = target - self.Q[state][action]
        self.Q[state][action] += self.step_size*delta
    
    def update_policy_on_end(self, state, action, reward):
        state = self.discretizer.discretize(state)
        delta = -self.Q[state][action]
        self.Q[state][action] += self.step_size*delta 


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

    epsilon = 0.05
    decay = 0.85
    discount = 1.0
    step_size = 1.0

    # agent = SarsaAgent(action_count, discount, step_size)
    agent = QLearningAgent(action_count, discount, step_size)


    for episode in range(5000):
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

        if episode % 100 == 0:
            agent.step_size *= decay
            agent.step_size = max(0.005, agent.step_size)
        
        if episode % 500 == 0:
            print (total_reward)

            show_episode(env, agent)