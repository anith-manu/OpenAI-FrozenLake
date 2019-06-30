import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv
import os, sys
from helpers import *


# Setup the parameters
problem_id = int(sys.argv[1])
reward_hole = 0.0
is_stochastic = True
max_episodes = 5000
max_iter_per_episode = 500

# Generate the environment
env = LochLomondEnv(problem_id=problem_id, is_stochastic=False,   reward_hole=reward_hole)

print(env.desc)

# Create a representation of the state space for use with AIMA A-star
state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

# Reset the random generator to a known state (for reproducability)
np.random.seed(12)

rewards = list()
episodes = list()

for e in range(max_episodes): # iterate over episodes
    observation = env.reset() # reset the state of the env to the starting state
    
    for iter in range(max_iter_per_episode):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action) # observe what happends when the action is taken
    
    
        print("e,iter,reward,done =" + str(e) + " " + str(iter)+ " " + str(reward)+ " " + str(done))
        
        # Check if we are done and monitor rewards etc...
        if(done and reward==reward_hole):
            env.render()
            print("We have reached a hole :-( [we can't move so stop trying; just give up]")
            break
                
        if (done and reward == +1.0):
            rewards.append(reward)
            episodes.append(e)
            print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
            break


# Print output to file
filename = "Random_Agent_Output " + str(problem_id)

outputGoals = "The agent reached the goal " + str(len(rewards))+ " number of times on episodes " + str(episodes)

outputNone = "The agent reached the goal O number of times."

with open(filename + ".txt", "w") as file:
    if(len(rewards) > 0):
        print(outputGoals)
        file.write(str(outputGoals))
    else:
        print(outputNone)
        file.write(str(outputNone))


def return_for_eval():
    return  max_episodes,env,max_iter_per_episode,observation_list,reward_list
