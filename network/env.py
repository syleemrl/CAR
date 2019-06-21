from utils import RunningMeanStd
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
		nan_count = 0
		
		self.sim_env.SetActions(actions)
		self.sim_env.Steps(False)

		for j in range(self.num_slaves):
			is_terminal, nan_occur, time_end = self.sim_env.IsNanAtTerminal(j)
			if not nan_occur:
				r = self.sim_env.GetRewardByParts(j)
				rewards.append(r)
				dones.append(is_terminal[j])
			else:
				rewards.append(None)
				dones.append(True)
				nan_count += 1
		
		states = self.sim_env.getStates()

		return states, np.array(rewards), dones, nan_count 
