from utils import RunningMeanStd
import numpy as np
import simEnv
import time
class Env(object):
	def __init__(self, original_ref, adaptive_ref, mode, num_slaves):
		self.num_slaves = num_slaves
		if mode =="adaptive":
			self.sim_env = simEnv.Env(num_slaves, "/motion/"+original_ref, "/network/output/"+adaptive_ref, mode)
		else:
			self.sim_env = simEnv.Env(num_slaves, "/motion/"+original_ref, "", mode)
		
		self.num_state = self.sim_env.GetNumState()
		self.num_action = self.sim_env.GetNumAction()

	def reset(self, i, b):
		self.sim_env.Reset(i, b)
	
	def stepForEval(self, action, i):
		self.sim_env.SetAction(action[0], i)
		self.sim_env.Steps()
		is_terminal, nan_occur, start, frame_elapsed, time_elapsed = self.sim_env.IsNanAtTerminal(i)
		r = self.sim_env.GetRewardByParts(i)

		state = self.sim_env.GetState(i)

		return state, r, is_terminal

	def step(self, actions):
		rewards = []
		dones = []
		frames = []
		times = []
		nan_count = 0
		
		self.sim_env.SetActions(actions)
		self.sim_env.Steps()
		for j in range(self.num_slaves):
			is_terminal, nan_occur, start, frame_elapsed, time_elapsed = self.sim_env.IsNanAtTerminal(j)
			if not nan_occur:
				r = self.sim_env.GetRewardByParts(j)
				rewards.append(r)
				dones.append(is_terminal)
				times.append(time_elapsed)
				frames.append(frame_elapsed)
			else:
				rewards.append([None])
				dones.append(True)
				times.append(time_elapsed)
				frames.append(frame_elapsed)
				nan_count += 1
		
		states = self.sim_env.GetStates()
		return states, rewards, dones, times, frames, nan_count 
