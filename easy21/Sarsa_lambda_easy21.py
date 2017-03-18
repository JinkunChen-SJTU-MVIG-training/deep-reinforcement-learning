import gym
import numpy as np
import sys
from easy21 import easy21Env
from collections import defaultdict

env = easy21Env()

def make_epsilon_greedy_policy(Q,epsilon,nA):
	def policy_fn(observation):
		A = np.ones(nA,dtype=float) * epsilon / nA
		best_action = np.argmax(Q[observation])
		A[best_action] += (1.0 - epsilon)
		return A
	return policy_fn

	
def sarsa_lambda_control_greedy(env,num_episodes,lamda,discount_factor=1.0,epsilon=0.1):
	returns_count_s = defaultdict(float)
	returns_count_sa = defaultdict(float)
	
	player_score_space = range(-9,32)
	dealer_score_space = range(-9,27)
	for x in player_score_space:
		for y in dealer_score_space:
			state_space = set((x,y))
	
	Q = defaultdict(lambda:np.zeros(env.action_space.n))
	
	policy = make_epsilon_greedy_policy(Q,epsilon,env.action_space.n)
	
	for i_episode in range(1,num_episodes + 1):
		E = defaultdict(lambda:np.zeros(env.action_space.n))
		state = env._reset()
		probs = policy(state)
		action = np.random.choice(np.arange(len(probs)),p=probs)
		
		while True:
			sa_pair = (state,action)
			next_state,reward,done,_ = env._step(action)
			if done:
				break
			
			probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(probs)),p=probs)
			td_error = reward + discount_factor * Q[next_state][next_action] - Q[state][action]
			E[state][action] += 1
			returns_count_s[state] += 1
			returns_count_sa[sa_pair] += 1
			epsilon = 100 / ( 100 + returns_count_s[state])
			
			for s in state_space:
				for a in range(2):
					enumerate_sa_pair = (s,a)
					Q[s][a] += td_error * E[s][a] / (1 if returns_count_sa[enumerate_sa_pair] == 0 else returns_count_sa[enumerate_sa_pair])
					E[s][a] *= discount_factor * lamda
			
			state = next_state
			action = next_action
			
	return Q,policy	

def print_observation(observation):
	player_score = observation[0]
	dealer_score = observation[1]
	print("Player score: {} , Dealer score: {}".format(player_score,dealer_score))

def random_action():
	action = np.random.choice([0,1])
	return action	

for n in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
	Q,policy = sarsa_lambda_control_greedy(env,num_episodes=50000,lamda=n)
	print("Training completed.")
	
	V = defaultdict(float)
	a = defaultdict(int)
	for state,actions in Q.items():
		action_value = np.max(actions)
		optimal_action = np.argmax(actions)
		V[state] = action_value
		a[state] = optimal_action
		
	#start optimal policy test
	win_times = 0
	total_times = 50000
	print("Test optimal policy started")
	for i in range(total_times+1):
		obb = env._reset()
		while True:
#			print_observation(obb)
			action = a[obb]
#			print("Taking action:{}".format(['stick','hit'][action]))
			obb,reward,done,_ = env._step(action)
			if done:
#				print_observation(obb)
#				print("Game end. Reward:{}".format(reward))
				if reward == 1:
					win_times += 1
				break

	accuracy = float(win_times) / float(total_times)
	print("The accuracy of Sarsa({}) control is {}".format(n,accuracy))
