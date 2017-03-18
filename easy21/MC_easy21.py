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
		

def mc_control_epsilon_greedy(env,num_episodes,discount_factor=1.0,epsilon=0.1):
	returns_sum = defaultdict(float)
	returns_count_s = defaultdict(float)
	returns_count_sa = defaultdict(float)
	
	Q = defaultdict(lambda:np.zeros(env.action_space.n))
	
	policy = make_epsilon_greedy_policy(Q,epsilon,env.action_space.n)
	
	for i_episode in range(1,num_episodes + 1):
		if i_episode % 1000 == 0:
			print("\rEpisode {}/{}.".format(i_episode,num_episodes))
			sys.stdout.flush()
			
		episode = []
		state = env._reset()
		for t in range(100):
			probs = policy(state)
			action = np.random.choice(np.arange(len(probs)),p=probs)
			next_state,reward,done,_ = env._step(action)
			episode.append((state,action,reward))
			if done:
				break
			state = next_state
		
		sa_in_episode = set((tuple(x[0]),x[1]) for x in episode)
		for state, action in sa_in_episode:
			sa_pair = (state, action)
			first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state and x[1] == action)
			G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
			
			returns_count_s[state] += 1.0
			returns_count_sa[sa_pair] += 1.0
			returns_sum[sa_pair] = G 
			Q[state][action] += (returns_sum[sa_pair] - Q[state][action]) / returns_count_sa[sa_pair]
			epsilon = 100 / ( 100 + returns_count_s[state] )
		
	return Q,policy

def print_observation(observation):
	player_score = observation[0]
	dealer_score = observation[1]
	print("Player score: {} , Dealer score: {}".format(player_score,dealer_score))

def random_action():
	action = np.random.choice([0,1])
	return action

Q, policy = mc_control_epsilon_greedy(env,num_episodes=50000,epsilon=0.1)

V = defaultdict(float)
a = defaultdict(int)
for state,actions in Q.items():
	action_value = np.max(actions)
	optimal_action = np.argmax(actions)
	V[state] = action_value
	a[state] = optimal_action

# start test optimal policy
win_times = 0
total_times = 50000
print("Test optimal policy started")
for i in range(total_times+1):
	obb = env._reset()
	while True:
#		print_observation(obb)
		action = a[obb]
#		print("Taking action:{}".format(['stick','hit'][action]))
		obb,reward,done,_ = env._step(action)
		if done:
#			print_observation(obb)
#			print("Game end. Reward:{}".format(reward))
			if reward == 1:
				win_times += 1
			break

accuracy = float(win_times) / float(total_times)
print("The accuracy of Monte-Carlo control is {}".format(accuracy))
