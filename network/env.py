from utils import RunningMeanStd
import numpy as np
import simEnv
import time
class Env(object):
	def __init__(self, motion, num_slaves):
		self.num_slaves = num_slaves
		self.motion = motion
		self.sim_env = simEnv.Env(num_slaves, motion)
		
		self.num_state = self.sim_env.GetNumState()
		self.num_action = self.sim_env.GetNumAction()

	def reset(self, i):
		self.sim_env.Reset(i, False)
	
	def step(self, actions):
		rewards = []
		dones = []
		times = []
		target_met_count = 0
		target_not_met_count = 0
		nan_count = 0
		
		self.sim_env.SetActions(actions)
		self.sim_env.Steps()
		for j in range(self.num_slaves):
			is_terminal, nan_occur, target_met, start, time_elapsed = self.sim_env.IsNanAtTerminal(j)
			if not nan_occur:
				r = self.sim_env.GetRewardByParts(j)
				rewards.append(r)
				dones.append(is_terminal)
				times.append(time_elapsed)
				if is_terminal:
					if target_met and time_elapsed > 68:
						target_met_count += 1
					else:
						target_not_met_count += 1
			else:
				rewards.append([None])
				dones.append(True)
				times.append(time_elapsed)
				nan_count += 1
		
		states = self.sim_env.GetStates()

		return states, rewards, dones, times, nan_count, target_met_count, target_not_met_count
