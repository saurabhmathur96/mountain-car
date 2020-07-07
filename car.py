import numpy as np

from collections import defaultdict
from functools import partial

import numpy as np
from tqdm import trange

bin_sizes = [7, 7]
ranges = [[-1.2, 0.6], [-0.07 , 0.07]]

def discretize_observation(observations, ranges, bin_sizes):
    discrete = []
    for observation, r, bin_size in zip(observations, ranges, bin_sizes):
        bins = np.linspace(*r, num=bin_size)
        discrete.append(np.digitize(observation, bins))

    return tuple(discrete)

def argmax_random(a):
    return np.random.choice(np.flatnonzero(a == a.max()))


def epsilon_greedy_policy(state, Q, epsilon, action_count):
    A = np.ones(action_count) * (epsilon / action_count)
    best = argmax_random(Q[state])
    A[best] += (1.0 - epsilon)
    return A

def sarsa(env, episode_count, discount=1.0, learning_rate=1.0):
    action_count = env.action_space.n

    Q = defaultdict(partial(np.zeros, (action_count,)))

    epsilon = 0.05
    decay = 0.85
    
    for episode in trange(episode_count):
        
        observation =  env.reset() # start state
        state = discretize_observation(observation, ranges, bin_sizes)
        
        
        A = epsilon_greedy_policy(state, Q, epsilon, action_count)
        action = np.random.multinomial(1, pvals=A).argmax() # first action
        
        alpha = max(0.005, learning_rate * (decay ** (episode//100)))
        total_reward = 0
         
        while True:  

            observation, reward, done, info = env.step(action) 
            next_state = discretize_observation(observation, ranges, bin_sizes) 

            total_reward += reward

            A = epsilon_greedy_policy(next_state, Q, epsilon, action_count)
            next_action = np.random.multinomial(1, pvals=A).argmax()
            
            target = reward + discount*Q[next_state][next_action]
            delta = target - Q[state][action]
            Q[state][action] += alpha*delta

            if done: break

            state = next_state
            action = next_action
        
        if episode % 50 == 0:
            print ('total reward  = ', total_reward)

    return Q


def qlearning(env, episode_count, discount=1.0, learning_rate=1.0):
    action_count = env.action_space.n

    Q = defaultdict(partial(np.zeros, (action_count,)))

    epsilon = 0.3
    decay = 0.85
    
    for episode in trange(episode_count):
        
        observation =  env.reset() # start state
        state = discretize_observation(observation, ranges, bin_sizes)
        
        
        A = epsilon_greedy_policy(state, Q, epsilon, action_count)
        action = np.random.multinomial(1, pvals=A).argmax() # first action
        
        alpha = max(0.005, learning_rate * (decay ** (episode//100)))
        total_reward = 0
         
        while True:  

            observation, reward, done, info = env.step(action) 
            next_state = discretize_observation(observation, ranges, bin_sizes) 

            total_reward += reward

            A = epsilon_greedy_policy(next_state, Q, epsilon, action_count)
            next_action = np.random.multinomial(1, pvals=A).argmax()
            
            target = reward + discount*Q[next_state].max()
            delta = target - Q[state][action]
            Q[state][action] += alpha*delta

            if done: break

            state = next_state
            action = next_action
        
        if episode % 50 == 0:
            print ('total reward  = ', total_reward)
            show_episode(env, Q)

    return Q


def show_episode(env, Q):
    action_count = env.action_space.n
    observation =  env.reset() # start state


    while  True:
        env.render()
        state = discretize_observation(observation, ranges, bin_sizes)
        
        action = Q[state].argmax()
        
        observation, reward, done, info = env.step(action) 
        
        
        if done: break


if __name__ == '__main__':
    import gym

    env = gym.make('MountainCar-v0')

    # Q = sarsa(env, episode_count=5000)
    Q = qlearning(env, episode_count=5000)
    
    show_episode(env, Q)


    import pickle
    pickle.dump(Q, open( "carQ.p", "wb" ))
    env.close()
