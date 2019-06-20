from trajectory_manager import TrajectoryManager
from utils import RunningMeanStd
import sim
import pickle
class Env(object):
	def __init__(self, motion, num_slaves):
		self.num_slaves = num_slaves
		self.motion = motion
		self.sim_env = sim.Env(motion, num_slaves, True, True, True)
		
		self.num_state = self.sim_env.GetNumState()
		self.num_action = self.sim_env.GetNumAction()

		self.trajectory_manager = TrajectoryManager(motion)
		traj, g_traj, index, t_index, timet = self.trajectory_manager.getTrajectory() 
		self.sim_env.SetReferenceTrajectories(len(traj), traj)
		self.sim_env.SetGoalTrajectories(len(traj), g_traj)

	def reset(self, i):
		t_index, timet = self.trajectory_manager.selectTime()
		self.sim_env.ResetWithTime(i, time)
	
	def step(self, actions):
		rewards = []
		dones = []
		nan_count = 0
		
		self.sim_env.SetActions(actions)
		self.sim_env.Steps(False)

		for j in range(self.num_slaves):
			is_terminal, nan_occur, time_end = self.sim_env.IsNanAtTerminal(j)
			if not nan_occur:
				r = self.Env.GetRewardByParts(j)
				rewards.append(r)
				dones.append(is_terminal[j])
			else:
				rewards.append(None)
				dones.append(True)
				nan_count += 1
		
		states = self.sim_env.getStates()

		return states, np.array(rewards), dones, nan_count 
