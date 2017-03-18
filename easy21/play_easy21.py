import numpy as np
import easy21
from easy21 import easy21Env

env = easy21Env()

def print_observation(observation):
    player_score = observation[0]
    dealer_score = observation[1]
    print("Player Score: {} , Dealer_score:{}".format(player_score,dealer_score))
    
def strategy():
    action = input("Please input your action(0 for stick and 1 for hit):")
    return action
	
print("Game started!")
observation = env._reset()
while True:
    print_observation(observation)
    action = strategy()
    print("Taking action:{}".format(action))
    observation,reward,done,_ = env._step(action)
    if done:
       print_observation(observation)
       print("Game end. Reward:{}" .format(reward))
       break
