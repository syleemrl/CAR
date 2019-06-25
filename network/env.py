from utils import RunningMeanStd
import numpy as np
import simEnv
import pickle
class Env(object):
	def __init__(self, motion, num_slaves):
		self.num_slaves = num_slaves
		self.motion = motion
		self.sim_env = simEnv.Env(num_slaves, motion)
		
		self.num_state = self.sim_env.GetNumState()
		self.num_action = self.sim_env.GetNumAction()

	def reset(self, i):
		self.sim_env.Reset(i, True)
	
	def step(self, actions):
		rewards = []
		dones = []
		time_ends = []
		nan_count = 0
		
		self.sim_env.SetActions(actions)
		self.sim_env.Steps()

		for j in range(self.num_slaves):
			is_terminal, nan_occur, time_end = self.sim_env.IsNanAtTerminal(j)
			if not nan_occur:
				r = self.sim_env.GetRewardByParts(j)
				rewards.append(r)
				dones.append(is_terminal)
				time_ends.append(time_end)
			else:
				rewards.append(None)
				dones.append(True)
				time_ends.append(time_end)
				nan_count += 1
		
		states = self.sim_env.GetStates()

		return states, np.array(rewards), dones, time_ends, nan_count 
