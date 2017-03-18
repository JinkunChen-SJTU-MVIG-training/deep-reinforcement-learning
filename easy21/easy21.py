import gym
from gym import spaces
from gym.utils import seeding

number = [1,2,3,4,5,6,7,8,9,10]
color = ['red','black','black']

def cmp(a,b):
    if a > b:
       return 1
    elif a == b:
       return 0
    else:
       return -1

def sum_hand(hand):
    Sum = 0
    for cards in hand:
        Sum += ((-1) if cards.color == 'red' else 1) * cards.number
    return Sum

def is_bust(hand):
    return sum_hand(hand) > 21 or sum_hand(hand) < 1

class card():
    def __init__(self,np_random,first=False):
	self.number = np_random.choice(number)
	if first:
	   self.color = 'black'
	else:
	   self.color = np_random.choice(color)

class easy21Env():
    def __init__(self):
        self.action_space = spaces.Discrete(2) #stick = 0, hit = 1
		
	self._seed()
		
	self._reset()
		
	self.nA = 2
	
    def _seed(self,seed=None):
	self.np_random, seed = seeding.np_random(seed)
	return [seed]
	
    def _reset(self):
        self.dealer = [card(self.np_random,True)]
        self.player = [card(self.np_random,True)]
        return self._get_obs()
		
    def _get_obs(self,done=False):
	if done:
            return (sum_hand(self.player),sum_hand(self.dealer))
        else:
            return (sum_hand(self.player),self.dealer[0].number)

    def _step(self,action):
        assert self.action_space.contains(action)
        if action==1: #hit
           self.player.append(card(self.np_random))
           if is_bust(self.player):
                done = True
                reward = -1
           else:
                done = False
                reward = 0	
        else: #stick
            done = True
            while(sum_hand(self.dealer) < 17):
                self.dealer.append(card(self.np_random))
		if is_bust(self.dealer):
                    reward = 1
                else:
                    reward = cmp(sum_hand(self.player),sum_hand(self.dealer))
        return self._get_obs(done),reward,done,{} 				
