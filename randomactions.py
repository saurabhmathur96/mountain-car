import gym
import numpy as np 



class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        
    def select_action(self):
        return self.action_space.sample()

env = gym.make('MountainCar-v0')
ob = env.reset()

agent = RandomAgent(env.action_space)


for _ in range(1000):
    env.render()
    # take a random action
    action = agent.select_action()
    ob, r, done, info = env.step(action)

    if done: break
env.close()
