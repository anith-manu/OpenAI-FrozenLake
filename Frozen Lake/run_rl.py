import gym
import numpy as np
import os, sys
import time
from uofgsocsai import LochLomondEnv


# Setup the parameters
problem_id = int(sys.argv[1])
reward_hole = -0.01
is_stochastic = True
max_episodes = 50000
max_iter_per_episode = 500
epsilon = 0.9
lr = 0.81
gamma = 0.96

# Generate Environment
env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole)

# Fill Q-table with 0's
Q_table = np.zeros((env.observation_space.n, env.action_space.n))


# Decide next step (Q-function)
def action_utility(state, epsilon, env, Q_table):
    step = 0
    if np.random.uniform(0, 1) < epsilon: # generate a number between 0 and 1 and see if it’s smaller than epsilon
        step = env.action_space.sample() #  If it’s smaller, then a random step is chosen using env.action_space.sample()
    else:
        step = np.argmax(Q_table[state, :]) # if it’s greater then we choose the step having the maximum value in the Q-table for state:
    return step


# Used to update Q_table
def learn(state, new_state, reward, action, Q_table, gamma, lr):
    predict = Q_table[state, action]
    target = reward + gamma * np.max(Q_table[new_state, :])
    Q_table[state, action] = Q_table[state, action] + lr * (target - predict)


iterations = []
for episode in range(max_episodes):
    state = env.reset()
    count = 0
    
    while count < max_iter_per_episode:
        env.render()
        action = action_utility(state, epsilon, env, Q_table) # Appropriate action is made
        new_state, reward, done, info = env.step(action)
        
        if (done and reward == +1.0):
            iterations.append(count)
        
        learn(state, new_state, reward, action, Q_table, gamma, lr) # Update Q-table
        state = new_state
        count += 1
        
        if done: # If episode is finished
            break

print(Q_table)


# Print output to file
filename = "RL_Agent_Output " + str(problem_id)
print(str(Q_table))
with open(filename + ".txt", "w") as file:
    file.write("State of Q_table \n")
    file.write(str(Q_table))


def return_iterations():
    return iterations
