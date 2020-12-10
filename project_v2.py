"""
Luis Berber
CSCI 166
December 11, 2020
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class GridWorld(object):
	def __init__(self, gridSize, items):
		self.stepReward = -1.0 # try different size penalties (-0.5)
		self.hLength = gridSize[0] # horizontal dimension
		self.vLength = gridSize[1] # vertical dimension
		self.gird = np.zeros(gridSize) # return array filled with zeros
		self.items = items
		
		self.stateSpace = list(range(self.hLength * self.vLength)) # list of states
		
		self.actionSpace = {'N': -self.hLength, 'S': self.hLength, 'E': 1, 'W': -1} # set of possible action values
		self.actions = ['N', 'S', 'E', 'W'] # list of possible actions
		
		self.S = self.int_S()
	
	def int_S(self):
		S = {}
		for state in self.stateSpace:
			for action in self.actions:
				reward = self.stepReward # accumulate movement penalty
				newState = state + self.actionSpace[action]
				
				# effect of each possible state
				if newState in self.items.get('trap').get('loc'): # enter trap
					reward += self.items.get('trap').get('reward')
				elif newState in self.items.get('trap2').get('loc'):
					reward += self.items.get('trap2').get('reward')
				elif newState in self.items.get('trap3').get('loc'):
					reward += self.items.get('trap3').get('reward')
				elif newState in self.items.get('trap4').get('loc'):
					reward += self.items.get('trap4').get('reward')
				elif newState in self.items.get('trap5').get('loc'):
					reward += self.items.get('trap5').get('reward')
				elif newState in self.items.get('goal').get('loc'): # enter goal
					reward += self.items.get('goal').get('reward')
				elif newState in self.items.get('goal2').get('loc'):
					reward += self.items.get('goal2').get('reward')
				elif newState in self.items.get('goal3').get('loc'):
					reward += self.items.get('goal3').get('reward')
				elif newState in self.items.get('randE').get('loc'): # enter random event
					gr = random.uniform(0, 1) # generate rand probability
					if 0 <= gr < 0.2: # move down 1 space if able
						if not self.checkMove(newState+self.vLength, newState):
							newState += self.vLength
					elif 0.2 <= gr < 0.8: # move up 1 space if able
						if not self.checkMove(newState-self.vLength, newState):
							newState -= self.vLength
					else: # move right 1 space if able
						if not self.checkMove(newState+1, newState):
							newState += 1
				elif newState in self.items.get('randE2').get('loc'):
					gr = random.uniform(0, 1)
					if 0 <= gr < 0.25:
						if not self.checkMove(newState-1, newState):
							newState -= 1
					elif 0.25 <= gr < 0.5:
						if not self.checkMove(newState+1, newState):
							newState += 1
					elif 0.5 <= gr < 0.75:
						if not self.checkMove(newState-self.vLength, newState):
							newState -= self.vLength
					else:
						if not self.checkMove(newState+self.vLength, newState):
							newState += self.vLength
				elif newState in self.items.get('randE3').get('loc'):
					gr = random.uniform(0, 1)
					if 0 <= gr < 0.15:
						if not self.checkMove(newState-1, newState):
							newState -= 1
					elif 0.15 <= gr < 0.5:
						if not self.checkMove(newState+1, newState):
							newState += 1
					elif 0.5 <= gr < 0.75:
						if not self.checkMove(newState-self.vLength, newState):
							newState -= self.vLength
					else:
						if not self.checkMove(newState+self.vLength, newState):
							newState += self.vLength
				elif self.checkMove(newState, state): # normal move
					newState = state # revert to old state
				
				S[(state, action)] = (newState, reward)
		return S
	
	def checkTerminal(self, state):
		return state in self.items.get('trap').get('loc') + self.items.get('trap2').get('loc') + self.items.get('trap3').get('loc') +\
		self.items.get('trap4').get('loc') + self.items.get('trap5').get('loc') +\
		self.items.get('goal').get('loc') + self.items.get('goal2').get('loc') + self.items.get('goal3').get('loc')
	
	def checkMove(self, newState, oldState):
		if newState not in self.stateSpace:
			return True
		elif oldState % self.hLength == 0 and newState % self.hLength == self.hLength - 1:
			return True
		elif oldState % self.hLength == self.hLength - 1 and newState % self.hLength == 0:
			return True
		else:
			return False
#end GridWorld

def displayArea(a, grid):
	a = np.reshape(a, (grid.vLength, grid.hLength)) # reshape an array without changing its data
	
	cmap = plt.cm.get_cmap('Greens', 100) # return the Colormap instance
	norm = plt.Normalize(a.min(), a.max()) # linearly normalizes data into the [0.0, 1.0] interval
	rgba = cmap(norm(a))
	
	# goals
	for g in grid.items.get('goal').get('loc'):
		index = np.unravel_index(g, a.shape) # converts a flat index or array of flat indices into a tuple of coordinate arrays
		rgba[index] = 0.0, 0.5, 0.8, 1.0
	for g in grid.items.get('goal2').get('loc'):
		index = np.unravel_index(g, a.shape)
		rgba[index] = 0.0, 0.5, 0.8, 1.0
	for g in grid.items.get('goal3').get('loc'):
		index = np.unravel_index(g, a.shape)
		rgba[index] = 0.0, 0.5, 0.8, 1.0
	# traps
	for t in grid.items.get('trap').get('loc'):
		index = np.unravel_index(t, a.shape)
		rgba[index] = 1.0, 0.5, 0.1, 1.0
	for t in grid.items.get('trap2').get('loc'):
		index = np.unravel_index(t, a.shape)
		rgba[index] = 1.0, 0.5, 0.1, 1.0
	for t in grid.items.get('trap3').get('loc'):
		index = np.unravel_index(t, a.shape)
		rgba[index] = 1.0, 0.5, 0.1, 1.0
	for t in grid.items.get('trap4').get('loc'):
		index = np.unravel_index(t, a.shape)
		rgba[index] = 1.0, 0.5, 0.1, 1.0
	for t in grid.items.get('trap5').get('loc'):
		index = np.unravel_index(t, a.shape)
		rgba[index] = 1.0, 0.5, 0.1, 1.0
	# random event (move)
	for r in grid.items.get('randE').get('loc'):
		index = np.unravel_index(r, a.shape)
		rgba[index] = 0.6, 0.5, 0.8, 1.0
	for r in grid.items.get('randE2').get('loc'):
		index = np.unravel_index(r, a.shape)
		rgba[index] = 0.6, 0.5, 0.8, 1.0
	for r in grid.items.get('randE3').get('loc'):
		index = np.unravel_index(r, a.shape)
		rgba[index] = 0.6, 0.5, 0.8, 1.0
	
	fig, ax = plt.subplots() # number of rows/columns of the subplot grid
	im = ax.imshow(rgba, interpolation = 'nearest') # display data as an image
	
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			c = 'w'
			if a[i, j] < 4:
				c = 'k'
			if a[i, j] != 0:
				text = ax.text(j, i, np.round(a[i, j], 2), ha = "center", va = "center", color = c) # add text to the axis
	plt.axis('off') # turn off axis lines and labels
	plt.show()
	
def displayPolicy(a, policy, grid):
	a = np.reshape(a, (grid.vLength, grid.hLength))
	policy = np.reshape(policy, (grid.vLength, grid.hLength))
	
	cmap = plt.cm.get_cmap('Greens', 10)
	norm = plt.Normalize(a.min(), a.max())
	rgba = cmap(norm(a))
	
	# goals
	for g in grid.items.get('goal').get('loc'):
		index = np.unravel_index(g, a.shape)
		rgba[index] = 0.0, 0.5, 0.8, 1.0
	for g in grid.items.get('goal2').get('loc'):
		index = np.unravel_index(g, a.shape)
		rgba[index] = 0.0, 0.5, 0.8, 1.0
	for g in grid.items.get('goal3').get('loc'):
		index = np.unravel_index(g, a.shape)
		rgba[index] = 0.0, 0.5, 0.8, 1.0
	# traps
	for t in grid.items.get('trap').get('loc'):
		index = np.unravel_index(t, a.shape)
		rgba[index] = 1.0, 0.5, 0.1, 1.0
	for t in grid.items.get('trap2').get('loc'):
		index = np.unravel_index(t, a.shape)
		rgba[index] = 1.0, 0.5, 0.1, 1.0
	for t in grid.items.get('trap3').get('loc'):
		index = np.unravel_index(t, a.shape)
		rgba[index] = 1.0, 0.5, 0.1, 1.0
	for t in grid.items.get('trap4').get('loc'):
		index = np.unravel_index(t, a.shape)
		rgba[index] = 1.0, 0.5, 0.1, 1.0
	for t in grid.items.get('trap5').get('loc'):
		index = np.unravel_index(t, a.shape)
		rgba[index] = 1.0, 0.5, 0.1, 1.0
	# random event (move)
	for r in grid.items.get('randE').get('loc'):
		index = np.unravel_index(r, a.shape)
		rgba[index] = 0.6, 0.5, 0.8, 1.0
	for r in grid.items.get('randE2').get('loc'):
		index = np.unravel_index(r, a.shape)
		rgba[index] = 0.6, 0.5, 0.8, 1.0
	for r in grid.items.get('randE3').get('loc'):
		index = np.unravel_index(r, a.shape)
		rgba[index] = 0.6, 0.5, 0.8, 1.0
	
	fig, ax = plt.subplots()
	im = ax.imshow(rgba, interpolation = 'nearest')
	
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			c = 'w'
			if a[i, j] < 4:
				c = 'k'
			if a[i, j] != 0:
				text = ax.text(j, i, policy[i, j], ha = "center", va = "center", color = c)
	plt.axis('off')
	plt.show()
	
def interateValues(grid, a, policy, gamma, theta, pStoch):
	converged = False
	i = 0
	sp = pStoch
	p = {'N': [sp + (1 - sp) / 4, (1 - sp) / 4, (1 - sp) / 4, (1 - sp) / 4],
		 'S': [(1 - sp) / 4, sp + (1 - sp) / 4, (1 - sp) / 4, (1 - sp) / 4],
		 'E': [(1 - sp) / 4, (1 - sp) / 4, sp + (1 - sp) / 4, (1 - sp) / 4],
		 'W': [(1 - sp) / 4, (1 - sp) / 4, (1 - sp) / 4, sp + (1 - sp) / 4]}
	while not converged:
		delta = 0
		for state in grid.stateSpace:
			i += 1
			if grid.checkTerminal(state):
				a[state] = 0
			else:
				oldA = a[state]
				newA = []
				for action in grid.actions:
					newAP = []
					for index, actionPol in enumerate(grid.actions):
						(newState, reward) = grid.S.get((state, actionPol))
						newAP.append(p.get(action)[index] * (reward + (gamma * a[newState])))
					newA.append(sum(newAP))
				a[state] = max(newA)
				delta = max(delta, np.abs(oldA - a[state]))
				converged = True if delta < theta else False
	
	for state in grid.stateSpace:
		i += 1
		newAS = []
		for action in grid.actions:
			(newState, reward) = grid.S.get((state, action))
			newAS.append(reward + gamma * a[newState])
		newAS = np.array(newAS)
		bestActionIndex = np.where(newAS == newAS.max())[0]
		policy[state] = grid.actions[bestActionIndex[0]]
	print(i, 'iterations through the state space')
	return a, policy

if __name__ == '__main__':
	gridSize = (8, 7)
	items = {'trap': {'reward': -10, 'loc': [30]},
			 'trap2': {'reward': -10, 'loc': [32]},
			 'trap3': {'reward': -10, 'loc': [47]},
			 'trap4': {'reward': -10, 'loc': [5]},
			 'trap5': {'reward': -10, 'loc': [50]},
			 'goal': {'reward': 10, 'loc': [4]},
	         'goal2': {'reward': 10, 'loc': [24]},
			 'goal3': {'reward': 10, 'loc': [52]},
			 'randE': {'reward': 0, 'loc': [16]},
			 'randE2': {'reward': 0, 'loc': [34]},
			 'randE3': {'reward': 0, 'loc': [46]}}
	# values to play around with
	gamma = 1.0
	theta = 1e-10
	pStoch = 0.7
	
	a = np.zeros(np.prod(gridSize)) # np.prod --> returns the product of array elements over a given axis
	policy = np.full(np.prod(gridSize), 'n') # np.full --> return a new array of given shape and type, filled with fill_value
	env = GridWorld(gridSize, items) # set up the grid world with the grid size and items
	a, policy = interateValues(env, a, policy, gamma, theta, pStoch)
	
	displayArea(a, env)
	displayPolicy(a, policy, env)

# end here
