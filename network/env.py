from utils import RunningMeanStd
import numpy as np
import simEnv
import time
from IPython import embed
class Env(object):
	def __init__(self, ref, directory, adaptive, parametric, num_slaves):
		self.num_slaves = num_slaves
		self.sim_env = simEnv.Env(num_slaves, "/motion/"+ref, directory, adaptive, parametric)
		
		self.num_state = self.sim_env.GetNumState()
		self.num_action = self.sim_env.GetNumAction()
		self.adaptive = adaptive

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
		terminal_reason = []
		nan_count = 0

		self.sim_env.SetActions(actions)
		self.sim_env.Steps()
		for j in range(self.num_slaves):
			is_terminal, nan_occur, start, frame_elapsed, time_elapsed, t = self.sim_env.IsNanAtTerminal(j)
			if not nan_occur:
				r = self.sim_env.GetRewardByParts(j)
				rewards.append(r)
				dones.append(is_terminal)
				times.append(time_elapsed)
				frames.append(frame_elapsed)
				terminal_reason.append(t)
			else:
				if self.adaptive:
					rewards.append([None, None])
				else:
					rewards.append([None])
				dones.append(True)
				times.append(time_elapsed)
				frames.append(frame_elapsed)
				terminal_reason.append(t)

				nan_count += 1
		states = self.sim_env.GetStates()
		return states, rewards, dones, times, frames, terminal_reason, nan_count 
